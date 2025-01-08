# Entity-Relation-Prediction_NLP

Relation Prediction on NYT29 entity-relation dataset.
APPROACH

### Step 1: Dataset analysis and cleaning

High quality data leads to better model training. The entity pairs and relations were
extracted from the given .tup files and the relation.txt files. The relations were mapped to
indices to form labels that would be one hot-encoded for the model. The extracted entity
pairs were mapped to the relations in a dictionary format based on the relations.txt and this
was done for every sentence in the .sent file. So, each sentence index had a dictionary and
that dictionary had the entity pairs as keys and the relations as the values. After formatting
and structuring the dataset, each sentence was decomposed into multiple duplicate
sentences each tagged with exactly one entity pair. This was done to ensure that the model
takes in only one entity pair per sample to predict its relations.

Eg: Word1 Word2 Word3 Word4 Word5 Word6

Entity pairs: {(Word2, Word 4), (Word3, Word4)
Sentences: Word1 \[E1\]Word2\[\E1\] Word3 \[E2\]Word4\[\E2\] Word5 Word6
Word1 Word2 \[E1\]Word3\[\E1\] \[E2\]Word4\[\E2\] Word5 Word6

Specifics:
Since the relation.txt file did not account for “other” entity for entity pairs that exist in the
sentence but do not have a defined relation, permutations of the entity pairs were
considered and those that do not belong to the set of relations defined were labeled as
“other”.

Since there were many pairs of relations that were defined with “other” label, it caused a
large class imbalance in the dataset which could bias model training to predict “other”
more often to get a low loss value with a higher probability. To deal with this, ONLY for the
training set, 20% of the “other” relation entity pairs were randomly sampled and kept while
constructing the train dataset. The test and validation class remained unmodified.

### Step 2: Model choice
Considered the BERT model, specifically the pretrained BERTForSequenceClassification
model “Bert-base-uncased” on huggingface (pretrained weights publicly available
) with multi-label classification head for this task. It can learn and represent semantics
robustly and is pretrained sufficiently to for entity relation understanding but performs
poorly unless finetuned for it.

### Step 3: Training
Performed all parameter finetuning using a normal training loop, calculating loss (in-built –
BCELogitsLoss for Binary Cross Entropy loss per label predicted).
The number of epochs was empirically chosen as epochs = 2 (Also considering resource
constraints).
Similarly, train/devbatch size was empirically chosen based on resources constraints,
where train/val batch_size = 4. Used the recommended optimizer (often used for BERT and
transformer training) : AdamW optimizer with initial learning rate as 2e-5, and default
momentum rates adam_beta1 as 0.9 and adam_beta2 as 0.999.
The best model was saved based on F1 score (as detailed in relevant papers).
Loss metric was the default loss metric for classification task: Binary Cross-entropy loss
(Logits Loss – per label).
At the end of each epoch, the model was evaluated, printing the loses, and the f1 score
obtained.


### Step 4: Evaluation
The same compute_metrics method was loaded with the test_dataloader to perform
evaluation and calculate the F1 score.
Training, Evaluation and Result: (Scores reported: F1 and weighted F1)
