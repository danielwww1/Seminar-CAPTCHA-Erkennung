# Seminar-CAPTCHA-Erkennung
Seminarprojekt zu "Moderne Deep Learning-Methoden sind in der Lage, CAPTCHAS und Antibot-Mechanismen erfolgreich zu lösen." von Daniel Weißer

Images & XMLs: https://drive.google.com/drive/folders/1mDJIC2UfkiIY7brclg3rcyk8AIoLKS7_?usp=sharing
Leider funktioniert das Model derzeit mit einigen Fehlern. Ich vermute sie treten aufgrund des kleinen Datensatzes auf. Die in der Seminarbenutzten Bilder wurden mithilfe eines gewichteten Modells, MobileNET, erstellt. 

alleklassenseminar.py erstellt ein ungefiltertes Modell, ohne Checkpoint function. Der trainingseminar.py erstellt ein Modell nur mit Klassen über 700 gelabelten Bildern mit Checkpoint.  
