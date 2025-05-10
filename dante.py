#!/usr/bin/env python3

import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
import wave
import pandas as pd

# === CONFIG ===
# UPDATE THIS to your Raspberry Pi’s static IP!
INTERFACE_IP = '192.168.4.157'         # <-- set your Pi's static IP here
MCAST_GRP = '239.69.85.204'            # Dante multicast group
MCAST_PORT = 5004                      # Usually 5004
SKIP_RTP_HEADER = True                 # True if RTP header (12 bytes) present
GAIN_FACTOR = 40.0                     # Adjust amplification factor

def main():
    print(f"🎧 Setting up socket on {INTERFACE_IP} for {MCAST_GRP}:{MCAST_PORT}")

    # === SETUP SOCKET ===
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', MCAST_PORT))
    mreq = struct.pack("4s4s", socket.inet_aton(MCAST_GRP), socket.inet_aton(INTERFACE_IP))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(5)

    print("🟡 Waiting for audio packets...")

    audio_samples = []
    packet_count = 0
    target_samples = 5000 * 48  # ~5 seconds of 48kHz audio

    try:
        while len(audio_samples) < target_samples:
            data, addr = sock.recvfrom(2048)
            packet_count += 1
            payload = data[12:] if SKIP_RTP_HEADER else data

            for i in range(0, len(payload), 3):
                if i + 2 >= len(payload):
                    break
                b = payload[i:i+3]
                prefix = b'\xff' if b[0] & 0x80 else b'\x00'
                sample = int.from_bytes(prefix + b, byteorder='big', signed=True)
                audio_samples.append(sample)

    except socket.timeout:
        print("⏱️ Timeout: Finished listening.")

    if len(audio_samples) == 0:
        print("❌ No audio captured. Exiting.")
        return

    print(f"✅ Captured {len(audio_samples)} samples from {packet_count} packets.")

    # === PROCESS ===
    audio_array = np.array(audio_samples, dtype=np.float32) / (2**23)
    amplified_audio = np.clip(audio_array * GAIN_FACTOR, -1.0, 1.0)

    print("🔎 First 10 raw samples:", audio_samples[:10])
    print("🔎 First 10 normalized (boosted):", amplified_audio[:10])

    # === SAVE CSV ===
    csv_filename = 'samples_preview_mono_boosted.csv'
    pd.DataFrame({
        'raw': audio_samples[:100],
        'norm_boosted': amplified_audio[:100]
    }).to_csv(csv_filename, index=False)
    print(f"📄 CSV saved: {csv_filename}")

    # === SAVE WAV ===
    wav_filename = 'captured_audio_mono_boosted.wav'
    with wave.open(wav_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)  # 24-bit samples packed as 3 bytes
        wf.setframerate(48000)
        frames = bytearray()
        for s in (amplified_audio * (2**23 - 1)).astype(np.int32):
            s_clipped = int(max(min(s, 8388607), -8388608))
            s_bytes = s_clipped.to_bytes(4, byteorder='big', signed=True)[1:]
            frames += s_bytes

        wf.writeframes(frames)

    print(f"📁 WAV saved: {wav_filename}")

    # === PLOT ===
    plt.figure(figsize=(12, 6))
    plt.plot(amplified_audio)
    plt.title('Captured Audio - Mono (Boosted)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
