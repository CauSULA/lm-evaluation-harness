import datasets

from lm_eval.base import Task, rf
from lm_eval.metrics import mean, matthews_corrcoef


class RussianSuperGlue(Task):
    VERSION = 0
    DATASET_PATH = "russian_super_glue"
    DATASET_NAME = "danetqa"

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
        return '{passage}\nВопрос: {question}\nОтвет (да/нет):'.format(**doc)

    def doc_to_target(self, doc):
        return ' да' if doc['label'] else ' нет'

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["passage"] + "\n" + doc["question"]

    def construct_requests(self, doc, ctx):
        return [rf.loglikelihood(ctx, e)[0] for e in [" да", " нет", " Да", " Нет"]]

    def process_results(self, doc, results):
        d = max(zip([" да", " нет", " Да", " Нет"], results), key=lambda x: x[1])
        print(d)
        pred = d[0].lower() == ' да'
        gold = doc["label"]
        print(gold, pred)
        return {"mcc": (gold, pred)}

    def aggregation(self):
        return {"mcc": matthews_corrcoef}

    def higher_is_better(self):
        return {"mcc": True}


# dataset = datasets.load_dataset("russian_super_glue", "danetqa")
# print(dataset)

# rsg = RussianSuperGlue()

