use brain::EmotionState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflexAction {
    Idle,
    Wave,
    Investigate,
    StepBack,
    AlertStance,
}

/// Avalia ações rápidas baseadas no vetor emocional VAD (roadmap Minecraft / Touhou Little Maid).
pub fn evaluate_emotional_reflex(emotion: &EmotionState) -> ReflexAction {
    if emotion.valence <= -0.5 && emotion.arousal >= 0.7 {
        ReflexAction::AlertStance
    } else if emotion.valence <= -0.4 && emotion.dominance < 0.4 {
        ReflexAction::StepBack
    } else if emotion.arousal >= 0.75 && emotion.valence >= 0.1 {
        ReflexAction::Investigate
    } else if emotion.valence >= 0.45 {
        ReflexAction::Wave
    } else {
        ReflexAction::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflex_evaluation() {
        let angry = EmotionState {
            valence: -0.7,
            arousal: 0.85,
            dominance: 0.6,
        };
        assert_eq!(evaluate_emotional_reflex(&angry), ReflexAction::AlertStance);

        let happy = EmotionState {
            valence: 0.6,
            arousal: 0.4,
            dominance: 0.6,
        };
        assert_eq!(evaluate_emotional_reflex(&happy), ReflexAction::Wave);
    }
}
