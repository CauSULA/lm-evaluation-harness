from lm_eval.base import Task, rf
from lm_eval.metrics import mean, matthews_corrcoef, f1_score

class OGE(Task):
    VERSION = 0
    DATASET_PATH = "batalovme/oge_prob"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return '{text}\nОтвет:'.format(**doc)

    def doc_to_target(self, doc):
        return ' ' + doc['answer']

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {'until': ["\n"]})

    def process_results(self, doc, results):
        pred = results[0]
        gold = doc["answer"]
        return {"acc": pred.strip() == gold.strip()}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}