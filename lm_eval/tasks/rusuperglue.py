import numpy as np

from lm_eval.base import Task, rf
from lm_eval.metrics import mean, matthews_corrcoef, f1_score

from itertools import chain


class DaNetQA(Task):
    VERSION = 0
    DATASET_PATH = "russian_super_glue"
    DATASET_NAME = "danetqa"
    ANSWERS = {
        True: [" да", " Да"],
        False: [" нет", " Нет"],
    }

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
        self._chain = list(chain(*self.ANSWERS.values()))
        return [rf.loglikelihood(ctx, e)[0] for e in self._chain]

    def process_results(self, doc, results):
        max_token = max(zip(self._chain, results), key=lambda x: x[1])
        for k, v in self.ANSWERS.items():
            if max_token[0] in v:
                pred = k
                break

        gold = doc["label"]
        print(max_token, gold, pred)
        return {"mcc": (gold, pred), "f1": (gold, pred), "acc": pred == gold}

    def aggregation(self):
        return {"mcc": matthews_corrcoef, "f1": f1_score, "acc": mean}

    def higher_is_better(self):
        return {"mcc": True, "f1": True, "acc": True}


class PARus(Task):
    VERSION = 0
    DATASET_PATH = "russian_super_glue"
    DATASET_NAME = "parus"

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
        # Drop the period
        connector = {
            "cause": "потому что",
            "effect": "поэтому",
        }[doc["question"]]
        return doc["premise"].strip()[:-1] + f", {connector}"

    def doc_to_target(self, doc):
        correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
        return " " + self.convert_choice(correct_choice)

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["passage"] + "\n" + doc["question"]

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]

    def construct_requests(self, doc, ctx):
        choice1 = " " + self.convert_choice(doc["choice1"])
        choice2 = " " + self.convert_choice(doc["choice2"])

        ll_choice1, _ = rf.loglikelihood(ctx, choice1)
        ll_choice2, _ = rf.loglikelihood(ctx, choice2)

        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)

        return {"mcc": (gold, pred), "f1": (gold, pred), "acc": pred == gold}

    def aggregation(self):
        return {"mcc": matthews_corrcoef, "f1": f1_score, "acc": mean}

    def higher_is_better(self):
        return {"mcc": True, "f1": True, "acc": True}



class RWSD(Task):
    VERSION = 0
    DATASET_PATH = "russian_super_glue"
    DATASET_NAME = "rwsd"
    ANSWERS = {
        True: [" да", " Да"],
        False: [" нет", " Нет"],
    }

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
        return '{text}\nОтносится ли фраза "{span1_text}" к фразе "{span2_text}"\nОтвет (да/нет):'.format(**doc)

    def doc_to_target(self, doc):
        return ' да' if doc['label'] else ' нет'

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["passage"] + "\n" + doc["question"]

    def construct_requests(self, doc, ctx):
        self._chain = list(chain(*self.ANSWERS.values()))
        return [rf.loglikelihood(ctx, e)[0] for e in self._chain]

    def process_results(self, doc, results):
        max_token = max(zip(self._chain, results), key=lambda x: x[1])
        for k, v in self.ANSWERS.items():
            if max_token[0] in v:
                pred = k
                break

        gold = doc["label"]
        print(max_token, gold, pred)
        return {"mcc": (gold, pred), "f1": (gold, pred), "acc": pred == gold}

    def aggregation(self):
        return {"mcc": matthews_corrcoef, "f1": f1_score, "acc": mean}

    def higher_is_better(self):
        return {"mcc": True, "f1": True, "acc": True}
