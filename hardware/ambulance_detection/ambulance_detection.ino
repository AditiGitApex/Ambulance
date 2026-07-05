/*
 * ambulance_detection.ino
 * 
 * ESP32 traffic light controller with FFT-based siren detection.
 *
 * Components:
 *   - W104 sound sensor AO  -> GPIO34
 *   - Red LED               -> GPIO18
 *   - Yellow LED            -> GPIO19
 *   - Green LED             -> GPIO21
 *
 * LED Priority Logic:
 *   - Camera detects ambulance  -> GREEN  (highest)
 *   - Siren frequency detected  -> YELLOW (mid)
 *   - Nothing detected          -> RED    (default)
 *
 * Serial Protocol (Python -> ESP32):
 *   'A' = ambulance detected this frame
 *   'N' = no ambulance this frame
 *
 * Siren detection:
 *   Sample W104 AO at 4000 Hz, run 256-point FFT,
 *   check dominant frequency in 500-1800 Hz band.
 */

#include <Arduino.h>
#include <arduinoFFT.h>

// ===== Pin definitions =====
#define SOUND_AO    34
#define LED_RED     18
#define LED_YELLOW  19
#define LED_GREEN   21

// ===== FFT parameters =====
#define SAMPLES         256
#define SAMPLING_FREQ   4000

const double samplingFreq = SAMPLING_FREQ;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLING_FREQ;

// Siren frequency band (typical siren fundamentals)
const double SIREN_FREQ_MIN = 500.0;
const double SIREN_FREQ_MAX = 1800.0;

// Amplitude threshold - tune this for your environment
// Higher = less sensitive (fewer false triggers)
double AMP_THRESHOLD = 800.0;

// FFT buffers
double vReal[SAMPLES];
double vImag[SAMPLES];
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, SAMPLES, samplingFreq);

// State
bool sirenDetected = false;
bool ambulanceDetected = false;
unsigned long lastAmbulanceMsg = 0;
const unsigned long AMBULANCE_TIMEOUT_MS = 1500;

void setup() {
    Serial.begin(115200);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(SOUND_AO, INPUT);

    digitalWrite(LED_RED, HIGH);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_GREEN, LOW);

    Serial.println("=================================");
    Serial.println("Ambulance Detection System Ready");
    Serial.println("Listening for siren (500-1800 Hz)");
    Serial.println("=================================");
}

void sampleAudio() {
    for (int i = 0; i < SAMPLES; i++) {
        unsigned long t0 = micros();
        vReal[i] = (double) analogRead(SOUND_AO);
        vImag[i] = 0.0;
        while (micros() - t0 < SAMPLE_INTERVAL_US) {
            // wait for next sample slot
        }
    }
}

bool detectSiren() {
    sampleAudio();

    // Remove DC offset
    double mean = 0;
    for (int i = 0; i < SAMPLES; i++) mean += vReal[i];
    mean /= SAMPLES;
    for (int i = 0; i < SAMPLES; i++) vReal[i] -= mean;

    // FFT
    FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
    FFT.compute(FFTDirection::Forward);
    FFT.complexToMagnitude();

    // Find peak in siren band
    int minBin = (int)(SIREN_FREQ_MIN * SAMPLES / samplingFreq);
    int maxBin = (int)(SIREN_FREQ_MAX * SAMPLES / samplingFreq);

    double peakMag = 0;
    double peakFreq = 0;
    for (int i = minBin; i <= maxBin; i++) {
        if (vReal[i] > peakMag) {
            peakMag = vReal[i];
            peakFreq = (double) i * samplingFreq / SAMPLES;
        }
    }

    if (peakMag > AMP_THRESHOLD) {
        Serial.print("[SIREN] freq=");
        Serial.print(peakFreq, 1);
        Serial.print(" Hz | mag=");
        Serial.println(peakMag, 0);
        return true;
    }
    return false;
}

void handleSerial() {
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == 'A') {
            ambulanceDetected = true;
            lastAmbulanceMsg = millis();
            Serial.println("[CAM] Ambulance detected");
        }
    }
    // Clear ambulance flag after timeout
    if (ambulanceDetected && (millis() - lastAmbulanceMsg > AMBULANCE_TIMEOUT_MS)) {
        ambulanceDetected = false;
        Serial.println("[CAM] Ambulance cleared");
    }
}

void updateLEDs() {
    
    if (ambulanceDetected && sirenDetected) {
        digitalWrite(LED_GREEN,  HIGH);
        digitalWrite(LED_YELLOW, HIGH);  // dono ON
        digitalWrite(LED_RED,    LOW);
    }
    else if (ambulanceDetected) {
        digitalWrite(LED_GREEN,  HIGH);
        digitalWrite(LED_YELLOW, LOW);
        digitalWrite(LED_RED,    LOW);
    } 
    else if (sirenDetected) {
        digitalWrite(LED_GREEN,  LOW);
        digitalWrite(LED_YELLOW, HIGH);
        digitalWrite(LED_RED,    LOW);
    } 
    else {
        digitalWrite(LED_GREEN,  LOW);
        digitalWrite(LED_YELLOW, LOW);
        digitalWrite(LED_RED,    HIGH);
    }
}

void loop() {
    sirenDetected = detectSiren();   // ~70ms (sample + FFT)
    handleSerial();                  // <1ms
    updateLEDs();                    // <1ms
    delay(20);                       // small pause for system stability
}