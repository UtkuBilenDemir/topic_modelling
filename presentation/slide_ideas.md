# SLIDE IDEAS

// TODO: Erkläre NLP

## 1. Topic Modelling

### 1.1. Was ist *topic modelling*
---
Ein *Machine Learning* (ML)-Verfahren in NLP, das die Mustern in den Wörter oder 
Ausdrücke in den Textsammlungen als *Topics* gruppiert.

* Jedes Dokument kann abhängig von der Methode ein einziges oder mehrere Topics haben.

* Meistens *unsupervised*: Eine vorherige Beschriftung/ Klassifikation der Dokumente 
ist nicht nötig (jedoch hilfreich).
---
Es ist grundsätzlich eine Methode, ein statistisches Modell um verschiedene
und möglichst auch latente Themen
in einer bestimmten Sammlung der Texte zu finden.

Topic Modelling gilt meistens als Überbegriff von den Methoden, 
die den Dokumente bestimmte Topics zuordnen. 

Hier eine kurze Anmerkung wegen der NLP-Terminologie. Die Sammlung von den Texten
wird **Corpus** und ein einzelner Text in diesem Corpus wird **Document** genannt.
Deswegen referiere ich sie in diesem Sinne ab jetzt.
<!--- Also ein Document kann jede Art des Textes sein --->

### 1.2. Wofür *topic modelling*

* Effektive Bearbeitung der großen Textsammlungen.

* Großes Potential verborgene Themen zu finden.

<!--- TODO: Expand and cut into pieces--->
Es gibt diverse Gründe *topic modelling* zu benutzen aber
der Grund für die Entwicklung (?)vom *topic modelling* war das Lesen zu unterstützen.
Die Leseunterstützung hier ist für die riesigen Corpora
mit einer großen Anzahl der Dokumente gemeint,
welche unmöglich für eine Person (oder für eine Gruppe) zu lesen sind.

### 1.3. SSH und *topic modelling*

Wie auch immer, unabhängig von den ersten Überlegungen haben die topic modelling
Methoden ziemlich schnell kommerzielle Anwendungsmöglichkeiten gefunden.
Einige von diesen Anwendungsszenarien sind z.B. die Analyse der Rezensionen,
spam-filter, chat-bots für die Webseiten zu schreiben usw.

Diese Methoden haben aber die bedeutenste Signifikanz
in den akademischen Bereichen bekommen. Z.B. in Bioinformatics, Chemie in
unterschiedlichen medizinischen Bereichen wird topic modelling immer populärer.

Jetzt kommen wir aber zu den Sozialwissenschaften.
Topic modelling ist bei den SoWis auch bei der Identifikation von den Variablen
und features nützlich geworden zusätlich zu den bisherigen Vorteilen.
Also, außer der Identifikation und Kategorisierung der Themen gibt es auch
einen Nutzen um direkt die Forschungsmethoden in SSH zu unterstützen.


## 2. Latent Dirichlet Allocation (LDA)

Wir werden heute eig. meist über LDA sprechen.

### 2.1. Was ist LDA

---

* **Die** *topic modelling* Methode.

* Findet durch die Gruppierungen der Wörter eine Verteilung der Topics in jedem
  Dokument.

* Entwickelt bei David Blei u.a. (2003).

* Analyse der Abstracts von den Publikationen ist eine von den ersten
  Anwendungen.

---

LDA ist eine von den topic modeling Annäherungen.
Schon vom Anfang der Geschichte von topic modelling LDA dominiert generell die
topic modelling Szene so stark, dass viele
eigentlich LDA meinen, wenn sie über topic modelling sprechen.  

Die erste Publikation über LDA ist von dem Informatiker David M. Blei u.a.
in 2003 geschrieben, in den späteren Publikationen wobei Blei LDA anwendet,
analysieren sie eigentlich die Abstracts von den akademischen
Publikationen um zu testen, ob sie unterschiedliche 
wiss. Gebieten von einander maschinell unterscheiden können.

https://web.archive.org/web/20120207011313/http://jmlr.csail.mit.edu/papers/volume3/blei03a/blei03a.pdf

https://www.researchgate.net/publication/262175901_Probabilistic_Topic_Models

### 2.2. Wie funktioniert LDA

---

* Annahme: Jedes *Document* ist eine Sammlung von den *Topics* 
(z.B. 60% Topic A, 30% Topic C, 10% Topic B).

* Am Anfang werden alle Wörter beliebig zu den *Topics* zugeordnet
(Die Anzahl der Topics `K` geben wir selbst ein).

* 
---

* LDA nimmt an, dass jedes Document eine Sammlung von den Topics ist. Das ergibt
eine VErteilung der Topics in jedem Document
(z.B. 60% Topic A, 30% Topic C, 10% Topic B).

* Am Anfang werden alle Wörter beliebig zu den Themen zugeordnet
(die Anzahl von den Themen (K) haben wir selbst eingegeben).

* Dann fängt die Iteration an und der Algorithmus geht jedes Wort und das assozierte Topic von dem Wort durch.
In jedem Zyklus gibt es einen zweiseitigen Prozess:

1. Wie oft das Topic in jedem Dokument auftritt
2. Wie oft wirklich das Wort in dem Topic überhaupt erscheint.

Ein Beispiel davon wäre: Der Algorithmus sieht,
* Dokument ist am Anfang beliebig zu dem Topic_B zugeordnet.
* Wort_1 befindet sich auch im Topic_B (Beliebig zugeordnet wieder am Anfang)
(1) dass Wort_1 wird eigentlich nicht so oft in dem Dokument auftreten und
Wörter von diesem Document auch nicht so oft in dem Topic_A auftreten ==>
Die Wörter vom Dokument  gehören wahr. in Topic_B.

Also der Algorithmus findet langsam eine Struktur, wenn dieser Prozess langsam konvergiert.
<!---
  - Auf einer Seite werden in jedes Topic dem charakteristischen Wörter zugeordnet,
  - Und auf der anderen Seite werden Dokumente gemäß ihren zu den einzelnen Topics relevanten Wörtern (proportional) zu den Topics zugeordnet. Wie gesagt haben sie aber nicht ein einziges Topic.
-->
<!--- Ich habe einige folgende Slides, wobei die innere Struktur weiter erklärt wird aber ich überspringe sie für jetzt und komme dann gerne zurück, wenn es relevante Fragen gibt.-->


<!--- TODO: ADD EXTRA SLIDES for LDA--->

<!--- TODO: Decide if the following belongs here or next chapter --->

* Also wie wir auf dem Bild sehen, sind alle Wahrscheinlichkeitsverteilungen. Die Dokumente sind Verteilungen von den unterschiedlichen Topics und die Topics sind Verteilungen von den Wörter.
<!--- TODO: Füge simple-LDA Vis. Beispiel--->

### 2.3. Was macht LDA/ topic modelling nicht

---
 * Topic modelling Methoden sind keine Alternativen zu der menschlichen Interpretation

 * LDA kann die Anzahl der Topics nicht bestimmen.


--- 




Vor allem sagen die topic modelling Methoden (mindestens diejenige,
die ich soweit gesehen habe) nicht,
wie viele Topics es in unserem Corpus gibt.
Es gibt Approximationsmethoden, welche wir gleich sehen werden
aber schlussendlich müssen wir selbst eintragen, wie viele Topics wir schätzen.

<!--- TODO: Was visuelles wäre cool--->

Also wir müssen für LDA-Anwendung folgende bestimmen
* DOcument-Term Matrix (Wir sehen gleich was das ist)
* Die Anzahl der Topics
* Die Anzahl der Iterationen (Wie wiele Zyklen sollte der Algorithmus durchführen)

## 3. Anwendungsbeispiel

Die weiteren Eigenschaften von LDA will ich mit einem Beispiel
weiterzeigen, damit wir gleichzeitig auch den Prozess sehen können.

### 3.1. Datensatz und die Herausforderung

Wir haben LDA erst beim KNoC-Projekt angewendet, also Knowledge Network on China heißt es.

Die Aufgabe war die wiss. Publikationen über China von den letzten 10 Jahren zu
analysieren und in unterschiedlichen wiss. Gebieten die Publikationen mit den
relevanten Themen zu identifizieren.

Was heißt *relevant* hier? Die Definition von der Relevanz war ziemlich breit in der Studie
<!--- TODO:Erkläre Besser --->
aber schlussendlich haben wir entschieden unterschiedliche wiss. Gebiete einzeln zu analysieren und dann die Relevanz von den gefundenen Topics zu bestimmen.

Wie auch immer war die Methode besonders neu für mich und bestimmte Sachen waren noch nicht klar und der Aufwand war unterschätzt.

Ich habe für eine beispielhafte Anwendung eine kleine Teilmenge von den präsenten wiss. Gebieten genommen. Diese könnt ihr unten sehen und den Datensatz damit von 32658 Publikationen auf [...] reduziert. Die Gebiete sind wie zu sehen sind nicht wesentlich unterschiedlich voneinander, sie sind relativ nahe Felder aber eine thematische Unterscheidung ist auch zu erwarten. Ich werde nun kurz die Schritte von der LDA-Anwendung auf unseren kleineren Datensatz zeigen, damit wir die möglichen Probleme und auch die Stärke von LDA sehen können.

<!--- TODO:Schreibe die Zahlen--->

### 3.2. Bag of words, Pre-Processing
* Vor allem funktioniert LDA mit dem Konzept *Bag of words*. Was damit gemeint ist, dass wir alle Dokumente nur als eine Sammlung von Wörtern beobachten. Also die Reihenfolge von den Wörtern bedeutet nichts, wir schauen nur auf unterschiedliche Wörter in dem Dokument und die Anzahl ihrer Erscheinungen. Hier haben wir z.B. eines von den Dokumenten. Was bedeutet ein Dokument in dem Kontext von KNoC? Das Abstract, die Überschrift und die Keywords. Alle sind zusammengemischt, weil Struktur uns nichts bedeutet und wenn bestimmte Wörter irgendwie z.B. in der Überschrift und dem Abstract wiederholt sind, ist es auch gut für uns, weil sie wahrscheinlich wichtige Wörter  sind und wir haben jetzt 2 Erscheinungen von denen.
<!--- TODO: Füge das Bild von einem Beispiel Dokument ein --->

* Damit wir diese Struktur haben können müssen wir zuerst alle Wörter mal trennen. Dieser Prozess heißt **Tokenization**. Das sehen wir schon auf dem Bild.
<!--- TODO: Füge das Bild der tokenizierten Liste ein --->

* Danach kommt die Lemmatization- oder Stemmingsprozess wobei alle Wörter von
ihren Suffixe usw. zu bereinigen und nur den Kern von dem Wort übrig zu lassen. Es gibt einen kleinen Unterschied zwischen den zwei Begriffen aber Lemmatization liefert bessere Resultate.
<!--- TODO: Füge das Bild der lemmatizationsliste ein --->

* Wie auch immer haben wir in den Dokumente noch immer viele Wörter, die in einem Bag of words nichts bedeuten werden. Z.B. die Artikel usw., diese heißen stop_words und wenn wir eine Liste von den stop_words haben, können wir sie von dem Text entfernen. Die Bilder erklären die Situation.
<!--- TODO: Füge das Bild der stop_words  ein --->

* Und letztens können wir optional auf die zusammen auftretenden Wörter schauen.
Da gibt es einen Begriff *n-grams* dafür, also unter bigrams und trigrams können wir oft zusammen erscheinende 2 oder 3 Wörter finden und sie als einzelne Wörter in unseren Dokumenten haben wie auf dem Beispiel. N-grams verleihen uns ein bisschen Struktur in unserem Bag of words.

<!--- TODO: BAG OF WORDS --->

### 3.3. Document-term matrix
<!--- DOCUMENT TERM MATRIX --->
* Das ist grundsätzlich eine Matrix, wobei die Zeilen die Dokumente rerpäsentieren,
* und die Spalten das Vokabular. Was ist mit dem Vokabular gemeint? Sie sind
alle Wörter, die überhaupt in unserem Corpus erscheinen.
* Die Zellen in dieser Matrix zeigen die Anzahl der Erscheinung jedes Wortes in den
einzelnen Dokumente.

<!---TODO: Visualize matrix--->
<!---TODO: Pre processing!!!--->

### 3.4. Das LDA-Modell
<!---Good of fitness, same as coherence estimation??--->
* Wir wählen die Anzahl der Topics selbst aus wie gesagt, deswegen, wenn wir nur
einfach schätzen wird unser Modell wahrscheinlich Falsch sein.

Also wir können beim Modell folgende bestimmen:
* Die Anzahl der Topics (K)
* Die Anzahl der Iterationen (Wie wiele Zyklen sollte der Algorithmus durchführen)
<!---TODO: NOCH???--->


### 3.5. *Coherence Estimation*

Wie gesagt, findet LDA nicht heraus, wie viele Themen es in unserem Corpus gibt aber
eine Approximation ist möglich. Sogenannte Coherence Estimation kann uns generell
zeigen wie KOnsisten der Unterschiedung von den Themen ist. Wenn wir in einer
bestimmten Spannweite von den Nummern für die Anzahl der Topics coherence estimation
testen, können wir generell schätzen, was für eine Zahl passend sein könnte.
<!---TODO: Visulization of the process--->

Das ist natürlich keinerlei ein PRozess, dem gesamten Entscheidung zu überlassen.
<!---Achtung: Personal bias is possible-->

<!---Nothing tops human interpretation, we are just equip it with the machine--->

### 3.6 LDA Visualization

Schauen wir jetzt kurz wie unsere Resultate aussehen. Diese Visualisierung finde ich besonders informativ, obwohl es nur eine Approximation ist. Auf der linken Seite sehen wir die Topics, je größer die Blasen sind, desto breiter das Topic. Diese sind aber keine Mengen, also eine DIstanz bedeutet nicht, dass die Topics nichts gemeinsames haben. Diese sind viel dimensionale Elemente und wir sehen nur eine Approximation auf einer 2 dimensionalen Ebene.

Auf der Rechten Seite sehen wir die Wörter in jedem Topic. Die blauen Balken zeigen wie oft die Wörter überhaupt in dem Corpus erscheinen und die roten Balken zeigen, wie oft die Wörter in diesem TOpic erscheinen.


<!---TODO: LDAvis--->
## 4. Die Signifikanz von *topic modelling*

### 4.1. Limitations
* Wenn wir gar keine Ahnung vom Kontext haben, bedeutet LDA nichts
* False positives, false negartives
* Word order doesn't matter but we are trying to overcome it with n-grams

### 4.2. Kritik
* Zu viel abhängig von den stemming/lemmatization
* Die MOdelle sind sensitiv zu den Iput-Data
* Hard to tell it is viable


### 4.3 Other methods

* Structural topic modelling : Sehr ähnlich wie LDA, benutzt aber metadata von den
Dokumente(Jahr, Ort usw.) und hilft der Forscherin besser zu interpretieren.
Entwickelt von einem Politikwissenschaftler.

Das ist ein R-Paket und macht auch die Pre-processing ziemlich einfacher.

One topic over time is possible with STM.

* Short text? stLDA-C (Tierney et al.). Für Tweets usw.


### 4.4. Kreative Nutzen

Ich werde die historischen thematischen Reflektionen über Pariser Kommune in der Linke analysieren.

## 5. Diskussion






---
## PHASES

1. PRe-process
   * Tokenization
   * Stop_words
   * Stemming/ Lemmatization
   * n-grams
2. Vocabulary
3. Matrix
4. Number of topics
5. Run the Model
6. Coherence estimation (Test continiously so we can get the optimal value)
7. Visualize coherence value, choose the best Model
8. Visualize the Model
