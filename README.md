# ACE Tennis AI – unabhängiger Schläger-Modellvergleich

Dieser Versuchszweig ist bewusst von `main` getrennt. Er verändert weder die veröffentlichte App noch deren IndexedDB-Daten, Videos oder Analysen.

## Ziel

Gesucht wird ein Modell, das einen echten Tennisschläger in privaten Trainingsvideos zuverlässig erkennt – seitlich und frontal, auch bei Bewegungsunschärfe und teilweiser Verdeckung – und anschließend auf dem iPhone lokal ausführbar ist.

## Ausgangslage

- Bestehender App-Stand auf `main`: Version 2.9.28 (Stand des Branch-Abzweigs).
- Bisheriges Modell: CourtSide YOLOv11n, ONNX, Klasse `racket`.
- Die bisherigen Tests in den ACE-Aufnahmen zeigen trotz Detektions- und Hand-/Arm-Tracking weiterhin deutliche Fehlzuordnungen.
- Die vorhandene Detektionspipeline bleibt in diesem Versuchszweig unverändert; neue Modelle werden zunächst separat bewertet.

## Kandidaten und Verfügbarkeit

| Kandidat | Bekannte Fakten | Eignung für den nächsten Test |
|---|---|---|
| CourtSide YOLOv11n (Bestand) | 2,6 Mio. Parameter, 640×640; vom App-Projekt bereits eingebunden. Das veröffentlichte Modellprofil nennt ca. 44 % Recall auf einem kleinen Validierungssatz (66 Bilder). | Referenz/Baseline |
| YOLO-Net (Du et al., 2026) | Tennis-spezifisches Modell mit 2,58 Mio. Parametern; im Paper 84,5 % Precision, 73,0 % Recall und 78,2 % mAP@0.5. Die Autoren stellen laut Data-Availability-Angabe nur Validierungsbilder und Annotationen öffentlich bereit; einen offiziellen Checkpoint/ONNX-Export habe ich in der Recherche nicht gefunden. | Fachlich sehr interessant, aber nicht sofort testbar; erst Autoren/Projektcode bzw. Gewichte beschaffen |
| YOLO-World v2 small | Open-Vocabulary-Modell; ONNX-Export ca. 48,8 MB/12,7 Mio. Parameter. Benötigt CLIP ViT-B/32 Text-Embeddings und NMS. Eine öffentliche iOS/CoreML-Referenzimplementierung mit getrenntem Detektor und Text-Encoder existiert; die gefundene ONNX-Variante ist AGPL-3.0 lizenziert. | Erster praktisch testbarer Alternativkandidat: lokal mit Prompt „tennis racket“ testen; Lizenz und iPhone-Laufzeit vor Integration prüfen |
| OWLv2 base | Zero-shot-Detektor; veröffentlichte Modellablage ca. 1,24 GB. | Für die erste iPhone-Testsituation zu groß; vorerst nicht priorisieren |
| YOLOv11 Racket Pose (Roboflow Universe) | Tennis-Schläger-Keypoint-Modell; öffentlich angegebene 1.100 Bilder, 88,0 % Precision, 79,4 % Recall und 84,4 % mAP@0.5. Der gefundene Eintrag ist ein gehostetes Roboflow-Modell; lokal ladbare Gewichte sind noch nicht bestätigt. | Beobachtungskandidat; nur weiterverfolgen, wenn ein lokaler Export ohne Cloud-Inferenz verfügbar ist |
| Grounding DINO | Open-set-Detektor mit Transformer-Backbone und separatem Text-Encoder. | Für lokale Echtzeit-/iPhone-Verarbeitung zunächst zu aufwendig; nicht erster Kandidat |

## Fairer Testablauf

1. Identische Testbilder und Videoframes für alle Modelle verwenden.
2. Mindestens folgende Fälle abdecken: Schlägerkopf klar sichtbar, Schläger seitlich/verkürzt, Schläger in Bewegung, Schläger vor dunkler/heller Kleidung, teilweise verdeckt, Schläger nicht vorhanden.
3. Pro Frame notieren: echter Schläger erkannt ja/nein, falscher Treffer ja/nein, Box überlappt den Schläger ausreichend, Konfidenz und Inferenzzeit.
4. Hauptkennzahlen: Recall (wie oft der echte Schläger gefunden wird), Precision (wie viele Treffer wirklich Schläger sind), Box-Qualität und Laufzeit auf dem iPhone.
5. Die Modellausgabe zunächst ohne Hand-/Arm-Heuristik vergleichen. Erst danach dieselbe Hand-/Arm-Zuordnung und denselben Tracker für die besten Modelle ergänzen.
6. Keine Videos oder Frames an externe Inferenzdienste senden. Gewichte dürfen von einem Modell-Host heruntergeladen werden; die Bildinferenz bleibt lokal auf dem Gerät.
7. Erst nach bestandenem Vergleich wird ein Kandidat in die reguläre App integriert.

## Vorläufige Entscheidung

YOLO-Net bleibt fachlich interessant, ist aber aktuell nicht direkt testbar: Die im Paper genannte Kaggle-Veröffentlichung umfasst ausdrücklich nur Validierungsbilder und Annotationen, keinen Modell-Checkpoint. Daher wird YOLO-World v2 small als erster praktisch ausführbarer Alternativkandidat priorisiert. Eine vorhandene iOS/CoreML-Referenz zeigt, dass Detektor und CLIP-Text-Encoder lokal auf iOS aufgeteilt werden können; das ist ein technischer Startpunkt, aber noch kein Leistungsnachweis auf ACE-Videos. Zusätzlich wird geprüft, ob das Roboflow-Racket-Pose-Modell lokal exportierbar ist. Alle Kandidaten werden vor einer App-Integration mit denselben ACE-Frames verglichen.

## Quellen

- Du et al. (2026), *YOLO-Net: A lightweight edge-enhanced detection model for small-object recognition in tennis match scenarios*, PLOS ONE: https://doi.org/10.1371/journal.pone.0335558
- Autoren-Datensatz (laut Paper nur Validierungsbilder und Annotationen): https://www.kaggle.com/datasets/taowang1123/yolo-net
- YOLO-World iOS/CoreML-Referenz (lokaler Detektor plus CLIP-Text-Encoder): https://github.com/S2dentik/yolo-world-ios
- Roboflow Tennis-Racket-Pose-Modell (Keypoint-Detektion, lokale Exportierbarkeit offen): https://universe.roboflow.com/mylab-4b02x/tennis-racket-pose
- YOLO-World v2 ONNX-Modellkarte und Input-Schema: https://huggingface.co/Instemic/yolo-world-onnx
- OWLv2 Modellkarte: https://huggingface.co/google/owlv2-base-patch16
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO

## Browser-Prototyp (isolierter Test)

Unter `experiments/racket-model-benchmark/index.html` liegt ein erster Browser-Prototyp. Er nimmt ein Video über die Dateiauswahl entgegen und verarbeitet die Frames lokal. Die Seite lädt CourtSide aus dem Repository, YOLO-World v2 Small von Hugging Face und den CLIP-Textencoder über Transformers.js. Die Videos selbst werden nicht an einen Server übertragen.

**Wichtiger Status:** Der Prototyp ist noch nicht auf einem iPhone verifiziert. Der Frame-Sprung wartet jetzt auf das tatsächliche `seeked`-Ereignis (mit Fehler bei Timeout), YOLO-Worlds Ausgabe wird anhand der Tensor-Dimensionen interpretiert und die beiden Detektoren laufen nacheinander, damit nicht beide ONNX-Modelle gleichzeitig im Speicher liegen. Der CLIP-Textencoder kann beim ersten Laden dennoch zusätzlichen Speicher und Datenvolumen benötigen. Die dargestellten Trefferzahlen sind Rohdetektionen; Precision/Recall erfordern zusätzlich manuell markierte Ground-Truth-Boxen.

Die Branch-Seite wird nicht automatisch unter GitHub Pages veröffentlicht, solange Pages auf den Hauptzweig zeigt. Die Dateien sind daher zunächst nur im Testzweig vorhanden; die ACE-Hauptapp bleibt unverändert.


## Eigenständige GitHub-Pages-Vorschau

Diese Kopie liegt im separaten Repository `Mrpomfi/ace-tennis-ai-benchmark`. Die Benchmark-Seite verwendet nur die unveränderten ONNX-Runtime- und CourtSide-Modell-Dateien aus dem Hauptrepository über öffentliche CDN-/Raw-URLs; das Video bleibt lokal im Browser. Die ACE-Haupt-App und ihre Pages-Einstellungen werden nicht verändert.
