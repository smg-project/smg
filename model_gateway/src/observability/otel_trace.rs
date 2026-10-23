//! OpenTelemetry tracing integration.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    },
    time::Duration,
};

use anyhow::Result;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    trace::{BatchConfigBuilder, BatchSpanProcessor, SdkTracer, SdkTracerProvider},
    Resource,
};
use tokio::task::spawn_blocking;
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};
use tracing::{Metadata, Subscriber};
use tracing_opentelemetry::{self, OpenTelemetrySpanExt};
use tracing_subscriber::{
    layer::{Context, Filter},
    Layer,
};

use super::events::get_module_path as events_module_path;

/// Whether OpenTelemetry tracing is enabled.
///
/// This flag guards access to TRACER and PROVIDER. We use Release/Acquire
/// ordering to ensure proper synchronization: writes to TRACER/PROVIDER
/// happen-before the Release store, and Acquire loads happen-before reads.
static ENABLED: AtomicBool = AtomicBool::new(false);
static TRACER: OnceLock<SdkTracer> = OnceLock::new();
static PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();
static ALLOWED_TARGETS: OnceLock<[&'static str; 3]> = OnceLock::new();

#[inline]
fn get_allowed_targets() -> &'static [&'static str; 3] {
    ALLOWED_TARGETS.get_or_init(|| {
        [
            "smg::otel-trace",
            "smg::observability::otel_trace",
            events_module_path(),
        ]
    })
}

/// Filter that only allows specific module targets to be exported to OTEL.
#[derive(Clone, Copy, Default)]
pub(crate) struct CustomOtelFilter;

impl CustomOtelFilter {
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    fn is_allowed(target: &str) -> bool {
        get_allowed_targets()
            .iter()
            .any(|allowed| target.starts_with(allowed))
    }
}

impl<S> Filter<S> for CustomOtelFilter
where
    S: Subscriber,
{
    #[inline]
    fn enabled(&self, meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        Self::is_allowed(meta.target())
    }

    #[inline]
    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> tracing::subscriber::Interest {
        if Self::is_allowed(meta.target()) {
            tracing::subscriber::Interest::always()
        } else {
            tracing::subscriber::Interest::never()
        }
    }
}

pub fn otel_tracing_init(enable: bool, otlp_endpoint: Option<&str>) -> Result<()> {
    if !enable {
        // Use Release to ensure any prior OTEL state changes are visible
        ENABLED.store(false, Ordering::Release);
        return Ok(());
    }

    let endpoint = otlp_endpoint.unwrap_or("localhost:4317");
    let endpoint = if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
        format!("http://{endpoint}")
    } else {
        endpoint.to_string()
    };

    global::set_text_map_propagator(TraceContextPropagator::new());

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(opentelemetry_otlp::Protocol::Grpc)
        .build()
        .map_err(|e| {
            // Logger may not be initialized yet during OTEL setup; use stderr
            #[expect(clippy::print_stderr)]
            {
                eprintln!("[tracing] Failed to create OTLP exporter: {e}");
            }
            anyhow::anyhow!("Failed to create OTLP exporter: {e}")
        })?;

    let batch_config = BatchConfigBuilder::default()
        .with_scheduled_delay(Duration::from_millis(500))
        .with_max_export_batch_size(64)
        .build();

    let span_processor = BatchSpanProcessor::builder(exporter)
        .with_batch_config(batch_config)
        .build();

    let resource = Resource::builder()
        .with_attribute(KeyValue::new("service.name", "smg"))
        .build();

    let provider = SdkTracerProvider::builder()
        .with_span_processor(span_processor)
        .with_resource(resource)
        .build();

    PROVIDER
        .set(provider.clone())
        .map_err(|_| anyhow::anyhow!("Provider already initialized"))?;

    let tracer = provider.tracer("smg");

    TRACER
        .set(tracer)
        .map_err(|_| anyhow::anyhow!("Tracer already initialized"))?;

    global::set_tracer_provider(provider);

    // Use Release ordering: all writes to TRACER/PROVIDER happen-before this store,
    // so any thread that loads ENABLED with Acquire will see the initialized state.
    ENABLED.store(true, Ordering::Release);

    // Logger may not be fully initialized yet during OTEL setup; use stderr
    #[expect(clippy::print_stderr)]
    {
        eprintln!("[tracing] OpenTelemetry initialized successfully");
    }
    Ok(())
}

/// Get the OpenTelemetry tracing layer. Must be called after `otel_tracing_init`.
pub fn get_otel_layer<S>() -> Result<Box<dyn Layer<S> + Send + Sync + 'static>>
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
{
    if !is_otel_enabled() {
        anyhow::bail!("OpenTelemetry is not enabled");
    }

    let tracer = TRACER
        .get()
        .ok_or_else(|| anyhow::anyhow!("Tracer not initialized. Call otel_tracing_init first."))?
        .clone();

    let layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(CustomOtelFilter::new());

    Ok(Box::new(layer))
}

/// Check if OpenTelemetry tracing is enabled.
///
/// Uses Acquire ordering to synchronize with the Release store in `otel_tracing_init`,
/// ensuring that if this returns true, TRACER and PROVIDER are fully initialized.
#[inline]
pub fn is_otel_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

pub async fn flush_spans_async() -> Result<()> {
    if !is_otel_enabled() {
        return Ok(());
    }

    let provider = PROVIDER
        .get()
        .ok_or_else(|| anyhow::anyhow!("Provider not initialized"))?
        .clone();

    spawn_blocking(move || provider.force_flush())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to join flush task: {e}"))?
        .map_err(|e| anyhow::anyhow!("Failed to flush spans: {e}"))?;

    Ok(())
}

/// The longest [`shutdown_otel`] waits for the spans still queued in the
/// batch exporter to reach the collector (the SDK's own default, written
/// here so the bound is known where it is paid). A collector that answers
/// costs a round trip; one that does not costs this much, once, and the
/// spans are lost as they would have been without the wait.
pub const OTEL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Flush the spans still queued and shut the tracer down, within
/// [`OTEL_SHUTDOWN_TIMEOUT`]; a no-op when tracing is off or already shut
/// down. Called on every exit of the process that initialised the tracer, a
/// clean stop and a failed start-up alike, while the runtime the exporter's
/// connection lives on is still up.
pub fn shutdown_otel() {
    // Use Acquire to ensure we see any prior OTEL operations
    if ENABLED.load(Ordering::Acquire) {
        // 0.32 removed `global::shutdown_tracer_provider`; shut down the stored
        // provider directly (its inner state is shared with the global provider).
        if let Some(provider) = PROVIDER.get() {
            if let Err(e) = provider.shutdown_with_timeout(OTEL_SHUTDOWN_TIMEOUT) {
                #[expect(clippy::print_stderr)]
                {
                    eprintln!("[tracing] Failed to shut down OpenTelemetry: {e}");
                }
            }
        }
        // Use Release to ensure shutdown completes before flag is cleared
        ENABLED.store(false, Ordering::Release);
        // Logger may already be shut down during process teardown; use stderr
        #[expect(clippy::print_stderr)]
        {
            eprintln!("[tracing] OpenTelemetry shut down");
        }
    }
}

/// Extract W3C trace context from incoming HTTP request headers.
///
/// Returns an [`opentelemetry::Context`] that can be set as the parent of a
/// `tracing::Span` via [`OpenTelemetrySpanExt::set_parent`], linking
/// server-side spans into the caller's distributed trace.
///
/// When OTel is disabled or no `traceparent` header is present the returned
/// context is empty, which is a no-op when passed to `set_parent`.
#[inline]
pub fn extract_trace_context_http(headers: &HeaderMap) -> opentelemetry::Context {
    if !is_otel_enabled() {
        return opentelemetry::Context::new();
    }

    struct HeaderExtractor<'a>(&'a HeaderMap);

    impl opentelemetry::propagation::Extractor for HeaderExtractor<'_> {
        #[inline]
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).and_then(|v| v.to_str().ok())
        }

        #[inline]
        fn keys(&self) -> Vec<&str> {
            self.0.keys().map(|k| k.as_str()).collect()
        }
    }

    global::get_text_map_propagator(|propagator| propagator.extract(&HeaderExtractor(headers)))
}

/// Inject W3C trace context headers into an HTTP request.
#[inline]
pub fn inject_trace_context_http(headers: &mut HeaderMap) {
    if !is_otel_enabled() {
        return;
    }

    let context = tracing::Span::current().context();

    struct HeaderInjector<'a>(&'a mut HeaderMap);

    impl opentelemetry::propagation::Injector for HeaderInjector<'_> {
        #[inline]
        fn set(&mut self, key: &str, value: String) {
            if let Ok(header_name) = HeaderName::from_bytes(key.as_bytes()) {
                if let Ok(header_value) = HeaderValue::from_str(&value) {
                    self.0.insert(header_name, header_value);
                }
            }
        }
    }

    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}

/// Inject W3C trace context into gRPC metadata.
#[inline]
pub fn inject_trace_context_grpc(metadata: &mut MetadataMap) {
    if !is_otel_enabled() {
        return;
    }

    let context = tracing::Span::current().context();

    struct MetadataInjector<'a>(&'a mut MetadataMap);

    impl opentelemetry::propagation::Injector for MetadataInjector<'_> {
        #[inline]
        fn set(&mut self, key: &str, value: String) {
            if let Ok(metadata_key) = MetadataKey::from_bytes(key.as_bytes()) {
                if let Ok(metadata_value) = MetadataValue::try_from(&value) {
                    self.0.insert(metadata_key, metadata_value);
                }
            }
        }
    }

    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut MetadataInjector(metadata));
    });
}

/// OpenTelemetry-based trace injector for gRPC clients.
///
/// This implements the `TraceInjector` trait from `smg_grpc_client`,
/// enabling distributed tracing across gRPC calls.
#[derive(Clone, Default)]
pub struct OtelTraceInjector;

impl smg_grpc_client::TraceInjector for OtelTraceInjector {
    fn inject(
        &self,
        metadata: &mut MetadataMap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        inject_trace_context_grpc(metadata);
        Ok(())
    }
}
