use super::*;

#[test]
fn test_softmax_basic() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = ClassifyResponseProcessingStage::softmax(&logits);

    // Probabilities should sum to 1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Highest logit should have highest probability
    assert!(probs[2] > probs[1]);
    assert!(probs[1] > probs[0]);
}

#[test]
fn test_softmax_empty() {
    let probs = ClassifyResponseProcessingStage::softmax(&[]);
    assert!(probs.is_empty());
}

#[test]
fn test_softmax_single() {
    let probs = ClassifyResponseProcessingStage::softmax(&[5.0]);
    assert_eq!(probs.len(), 1);
    assert!((probs[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_numerical_stability() {
    // Large values that would overflow without max subtraction
    let logits = vec![1000.0, 1001.0, 1002.0];
    let probs = ClassifyResponseProcessingStage::softmax(&logits);

    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(probs[2] > probs[1]);
}

#[test]
fn test_argmax() {
    assert_eq!(ClassifyResponseProcessingStage::argmax(&[0.1, 0.7, 0.2]), 1);
    assert_eq!(
        ClassifyResponseProcessingStage::argmax(&[0.9, 0.05, 0.05]),
        0
    );
    assert_eq!(ClassifyResponseProcessingStage::argmax(&[0.1, 0.1, 0.8]), 2);
}

#[test]
fn test_get_label_with_mapping() {
    let mut id2label = HashMap::new();
    id2label.insert(0, "negative".to_string());
    id2label.insert(1, "positive".to_string());

    assert_eq!(
        ClassifyResponseProcessingStage::get_label(Some(&id2label), 0),
        "negative"
    );
    assert_eq!(
        ClassifyResponseProcessingStage::get_label(Some(&id2label), 1),
        "positive"
    );
    assert_eq!(
        ClassifyResponseProcessingStage::get_label(Some(&id2label), 2),
        "LABEL_2"
    ); // Fallback for unknown
}

#[test]
fn test_get_label_without_mapping() {
    let id2label = HashMap::new();
    assert_eq!(
        ClassifyResponseProcessingStage::get_label(Some(&id2label), 0),
        "LABEL_0"
    );
    assert_eq!(
        ClassifyResponseProcessingStage::get_label(Some(&id2label), 5),
        "LABEL_5"
    );
    // No worker / no primary card
    assert_eq!(
        ClassifyResponseProcessingStage::get_label(None, 0),
        "LABEL_0"
    );
}
