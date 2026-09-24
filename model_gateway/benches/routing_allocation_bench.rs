use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use openai_protocol::{
    common::GenerationRequest,
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseReasoningContent,
        ResponsesRequest, StringOrContentParts,
    },
};

fn extract_text_for_routing_old(req: &ResponsesRequest) -> String {
    match &req.input {
        ResponseInput::Text(text) => text.clone(),
        ResponseInput::Items(items) => items
            .iter()
            .filter_map(|item| match item {
                ResponseInputOutputItem::Message { content, .. } => {
                    let texts: Vec<String> = content
                        .iter()
                        .filter_map(|part| match part {
                            ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                            ResponseContentPart::InputText { text } => Some(text.clone()),
                            ResponseContentPart::InputImage { .. }
                            | ResponseContentPart::InputFile { .. }
                            | ResponseContentPart::Refusal { .. } => None,
                        })
                        .collect();
                    if texts.is_empty() {
                        None
                    } else {
                        Some(texts.join(" "))
                    }
                }
                ResponseInputOutputItem::SimpleInputMessage { content, .. } => match content {
                    StringOrContentParts::String(s) => Some(s.clone()),
                    StringOrContentParts::Array(parts) => {
                        let texts: Vec<String> = parts
                            .iter()
                            .filter_map(|part| match part {
                                ResponseContentPart::InputText { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect();
                        if texts.is_empty() {
                            None
                        } else {
                            Some(texts.join(" "))
                        }
                    }
                },
                ResponseInputOutputItem::Reasoning { content, .. } => {
                    let texts: Vec<String> = content
                        .iter()
                        .map(|part| match part {
                            ResponseReasoningContent::ReasoningText { text } => text.clone(),
                        })
                        .collect();
                    if texts.is_empty() {
                        None
                    } else {
                        Some(texts.join(" "))
                    }
                }
                ResponseInputOutputItem::FunctionToolCall { arguments, .. } => {
                    Some(arguments.clone())
                }
                ResponseInputOutputItem::FunctionCallOutput { output, .. } => {
                    output.to_text_only().ok()
                }
                ResponseInputOutputItem::McpApprovalRequest { .. } => None,
                ResponseInputOutputItem::McpApprovalResponse { .. } => None,
                ResponseInputOutputItem::ImageGenerationCall { .. } => None,
                ResponseInputOutputItem::Compaction { .. } => None,
                ResponseInputOutputItem::ComputerCall { .. } => None,
                ResponseInputOutputItem::ComputerCallOutput { .. } => None,
                ResponseInputOutputItem::CustomToolCall { .. }
                | ResponseInputOutputItem::CustomToolCallOutput { .. } => None,
                ResponseInputOutputItem::ShellCall { .. }
                | ResponseInputOutputItem::ShellCallOutput { .. } => None,
                ResponseInputOutputItem::ItemReference { .. } => None,
                ResponseInputOutputItem::ApplyPatchCall { .. }
                | ResponseInputOutputItem::ApplyPatchCallOutput { .. } => None,
                // T5 schema-only: forced-cascade arm, no behavior.
                ResponseInputOutputItem::LocalShellCall { .. }
                | ResponseInputOutputItem::LocalShellCallOutput { .. } => None,
                // T11 schema-only: forced-cascade arm, no behavior.
                ResponseInputOutputItem::McpCall { .. }
                | ResponseInputOutputItem::McpListTools { .. } => None,
            })
            .collect::<Vec<String>>()
            .join(" "),
    }
}

fn extract_text_for_routing_new(req: &ResponsesRequest) -> String {
    req.extract_text_for_routing()
}

fn create_bench_request() -> ResponsesRequest {
    let mut items = Vec::new();

    // Create 10 message items, each with 10 inner text chunks
    for i in 0..10 {
        let mut content = Vec::new();
        for j in 0..10 {
            content.push(ResponseContentPart::InputText {
                text: format!("word_{i}_{j}"),
            });
        }
        items.push(ResponseInputOutputItem::Message {
            id: format!("msg_{i}"),
            role: "user".to_string(),
            content,
            status: None,
            phase: None,
        });
    }

    ResponsesRequest {
        input: ResponseInput::Items(items),
        model: "test-model".to_string(),
        ..Default::default()
    }
}

fn routing_allocation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_allocation");
    let req = create_bench_request();

    group.bench_with_input(
        BenchmarkId::new("old_O_N_alloc", "10_items"),
        &req,
        |b, r| b.iter(|| extract_text_for_routing_old(black_box(r))),
    );

    group.bench_with_input(
        BenchmarkId::new("new_O_1_alloc", "10_items"),
        &req,
        |b, r| b.iter(|| extract_text_for_routing_new(black_box(r))),
    );

    group.finish();
}

criterion_group!(benches, routing_allocation_benchmark);
criterion_main!(benches);
