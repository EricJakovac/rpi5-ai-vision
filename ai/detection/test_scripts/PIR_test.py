import RPi.GPIO as GPIO
import time

PIR_PIN = 17
COOLDOWN = 30        # sekundi između detekcija
MIN_TRIGGER_TIME = 0.8  # ignorira signale (filtrira šum)

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

print("⏳ Stabilizacija senzora (60s)...")
time.sleep(60)
print("✅ Senzor spreman!\n")

last_trigger = 0
trigger_start = None
last_debug_state = -1

try:
    while True:
        state = GPIO.input(PIR_PIN)
        now = time.time()

        # Debug: ispiši svaki put kad se stanje promijeni
        if state != last_debug_state:
            print(f"[DEBUG] GPIO pin {PIR_PIN} = {state} ({'HIGH - pokret!' if state == 1 else 'LOW - mirovanje'})")
            last_debug_state = state

        if state == 1:
            if trigger_start is None:
                trigger_start = now
            
            # Signal traje dovoljno dugo i prošlo je cooldown vrijeme
            elif (now - trigger_start >= MIN_TRIGGER_TIME and 
                  now - last_trigger >= COOLDOWN):
                print(f"🟢 Pokret detektiran! (trajanje signala: {now-trigger_start:.1f}s)")
                print("   → Pokrecem AI detekciju...")
                last_trigger = now
                trigger_start = None

        else:
            if trigger_start is not None:
                duration = now - trigger_start
                if duration < MIN_TRIGGER_TIME:
                    print(f"⚪ Ignoriram kratki signal ({duration:.2f}s) – vjerojatno šum")
            trigger_start = None

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Gašenje...")
    GPIO.cleanup()