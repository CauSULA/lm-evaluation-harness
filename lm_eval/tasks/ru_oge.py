from lm_eval.base import Task, rf
from lm_eval.metrics import mean, matthews_corrcoef, f1_score

from itertools import chain


class OGE_math(Task):
    VERSION = 0
    DATASET_PATH = "batalovme/RussianExams"
    DATASET_NAME = "math_tasks"

    def has_training_docs(self):
        return False

    def has_test_docs(self):
        return False
    
    def has_validation_docs(self):
        return True
    
    def validation_docs(self):
        return self.dataset["test"]

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


class OGE_math_yes_no(Task):
    VERSION = 0
    DATASET_PATH = "batalovme/RussianExams"
    DATASET_NAME = "yes_no_math_tasks"
    ANSWERS = {
        True: [" да", " Да"],
        False: [" нет", " Нет"],
    }

    def has_training_docs(self):
        return False

    def has_test_docs(self):
        return False
    
    def has_validation_docs(self):
        return True
    
    def validation_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return '{statement}\nОтвет (да/нет):'.format(**doc)

    def doc_to_target(self, doc):
        return ' да' if doc['label'] else ' нет'

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["statement"]

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
        return {"mcc": (gold, pred), "f1": (gold, pred), "acc": pred == gold}

    def aggregation(self):
        return {"mcc": matthews_corrcoef, "f1": f1_score, "acc": mean}

    def higher_is_better(self):
        return {"mcc": True, "f1": True, "acc": True}


class OGE_rus_basis_yes_no(Task):
    VERSION = 0
    DATASET_PATH = "batalovme/RussianExams"
    DATASET_NAME = "russian_basis_tasks"
    ANSWERS = {
        True: [" да", " Да"],
        False: [" нет", " Нет"],
    }

    def has_training_docs(self):
        return False

    def has_test_docs(self):
        return False
    
    def has_validation_docs(self):
        return True
    
    def validation_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return 'Верно ли что грамматическими членами (основой) предложения "{sentence}" является "{basis}"\nОтвет (да/нет):'.format(**doc)

    def doc_to_target(self, doc):
        return ' да' if doc['label'] else ' нет'

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["sentence"] + ' ' + doc["basis"]

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
        return {"mcc": (gold, pred), "f1": (gold, pred), "acc": pred == gold}

    def aggregation(self):
        return {"mcc": matthews_corrcoef, "f1": f1_score, "acc": mean}

    def higher_is_better(self):
        return {"mcc": True, "f1": True, "acc": True}


class OGE_rus_phrase_conn(Task):
    VERSION = 0
    DATASET_PATH = "batalovme/RussianExams"
    DATASET_NAME = "russian_phrase_conn_tasks"

    def has_training_docs(self):
        return False

    def has_test_docs(self):
        return False
    
    def has_validation_docs(self):
        return True
    
    def validation_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return 'Замените словосочетание "{phrase}" синонимичным словосочетанием со связью {connection}\nОтвет:'.format(**doc)

    def doc_to_target(self, doc):
        return ' ' + doc['answer']

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["phrase"] + ' ' + doc["connection"]

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
