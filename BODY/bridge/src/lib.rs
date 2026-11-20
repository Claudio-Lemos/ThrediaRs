use serde::{Deserialize, Serialize};

/// Pacotes de protocolo bidirecional Rust ↔ Java (Fabric / Touhou Little Maid — Roadmap).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum BridgePacket {
    Ping { timestamp_ms: u64 },
    WorldEvent { event: String, delta_valence: f32, delta_arousal: f32 },
    NavigateTo { x: f64, y: f64, z: f64, dimension: String },
    PlayAnimation { name: String },
}

pub fn serialize_packet(packet: &BridgePacket) -> Result<String, serde_json::Error> {
    serde_json::to_string(packet)
}

pub fn deserialize_packet(raw: &str) -> Result<BridgePacket, serde_json::Error> {
    serde_json::from_str(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_packet_roundtrip() {
        let pkt = BridgePacket::NavigateTo {
            x: 10.5,
            y: 64.0,
            z: -22.0,
            dimension: "minecraft:overworld".to_string(),
        };
        let json = serialize_packet(&pkt).unwrap();
        let decoded = deserialize_packet(&json).unwrap();
        assert_eq!(pkt, decoded);
    }
}
