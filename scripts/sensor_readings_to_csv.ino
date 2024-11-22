#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <BH1750.h>

// Create instances of the sensors
Adafruit_BME280 bme;  // BME280 instance
BH1750 lightMeter;    // BH1750 instance

// CSV data buffer
String csvData = "Timestamp (ms),Temperature (°C),Humidity (%),Pressure (hPa),Light Level (lux)\n";

// Timer variables
unsigned long startTime;
unsigned long currentTime;

void setup() {
  // Initialize Serial Monitor
  Serial.begin(9600);
  Serial.println(F("BME280 and BH1750 Data Logging to CSV"));

  // Initialize BME280
  if (!bme.begin(0x76)) {  // Default I2C address for BME280
    Serial.println(F("Error initializing BME280! Check wiring."));
    while (1)
      ;
  }
  Serial.println(F("BME280 Initialized"));

  // Initialize BH1750
  if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    Serial.println(F("Error initializing BH1750! Check wiring."));
    while (1)
      ;
  }
  Serial.println(F("BH1750 Initialized"));

  // Set start time for data logging
  startTime = millis();
}

void loop() {
  // Get current time
  currentTime = millis();

  // If 30 seconds have passed, export CSV and terminate
  if (currentTime - startTime > 30000) {
    Serial.println(F("\nData Logging Complete! Exporting CSV:\n"));
    Serial.println(csvData);
    while (1)
      ; // Stop further execution
  }

  // Read sensor values
  float temperature = bme.readTemperature();
  float humidity = bme.readHumidity();
  float pressure = bme.readPressure() / 100.0F;
  float lightLevel = lightMeter.readLightLevel();

  // Get timestamp
  unsigned long timestamp = millis();

  // Append readings to CSV data
  csvData += String(timestamp) + ",";
  csvData += String(temperature) + ",";
  csvData += String(humidity) + ",";
  csvData += String(pressure) + ",";
  if (lightLevel < 0) {
    csvData += "Error\n";
  } else {
    csvData += String(lightLevel) + "\n";
  }

  // Print readings to Serial Monitor for reference
  Serial.println(F("\n----------------------------------"));
  Serial.println(F("Sensor Readings:"));
  Serial.print(F("Timestamp: "));
  Serial.println(timestamp);
  Serial.print(F("Temperature: "));
  Serial.println(temperature);
  Serial.print(F("Humidity: "));
  Serial.println(humidity);
  Serial.print(F("Pressure: "));
  Serial.println(pressure);
  if (lightLevel < 0) {
    Serial.println(F("Error reading light level!"));
  } else {
    Serial.print(F("Light Level: "));
    Serial.println(lightLevel);
  }

  // Wait for 1 second before the next reading
  delay(1000);
}