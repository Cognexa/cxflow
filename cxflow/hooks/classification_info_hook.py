from .abstract_hook import AbstractHook

from sklearn.metrics import precision_recall_fscore_support


class ClassificationInfoHook(AbstractHook):
    def __init__(self, predicted_variable: str, gold_variable: str, f1_average: str, **kwargs):
        super().__init__(**kwargs)

        self.predicted_variable = predicted_variable
        self.gold_variable = gold_variable
        self.f1_average = f1_average

        self._reset()

    def _reset(self):
        self.train_predicted = []
        self.train_gold = []
        self.valid_predicted = []
        self.valid_gold = []
        self.test_predicted = []
        self.test_gold = []

    def after_batch(self, stream_type: str, results: dict, **kwargs) -> None:
        if stream_type == 'train':
            self.train_predicted += list(results[self.predicted_variable])
            self.train_gold += list(results[self.gold_variable])
        elif stream_type == 'valid':
            self.valid_predicted += list(results[self.predicted_variable])
            self.valid_gold += list(results[self.gold_variable])
        elif stream_type == 'test':
            self.test_predicted += list(results[self.predicted_variable])
            self.test_gold += list(results[self.gold_variable])
        else:
            raise ValueError('stream_type must be either train, valid or test')

    def before_first_epoch(self, valid_results: dict, test_results: dict=None, **kwargs) -> None:
        valid_precision, valid_recall, valid_fscore, _ = self._get_results(self.valid_gold, self.valid_predicted)
        valid_results['precision'] = valid_precision
        valid_results['recall'] = valid_recall
        valid_results['fscore'] = valid_fscore

        test_precision, test_recall, test_fscore, _ = self._get_results(self.test_gold, self.test_predicted)
        test_results['precision'] = test_precision
        test_results['recall'] = test_recall
        test_results['fscore'] = test_fscore

        self._reset()

    def after_epoch(self, epoch_id: int, train_results: dict, valid_results: dict, test_results: dict=None,
                    **kwargs) -> None:

        train_precision, train_recall, train_fscore, _ = self._get_results(self.train_gold, self.train_predicted)
        train_results['precision'] = train_precision
        train_results['recall'] = train_recall
        train_results['fscore'] = train_fscore

        valid_precision, valid_recall, valid_fscore, _ = self._get_results(self.valid_gold, self.valid_predicted)
        valid_results['precision'] = valid_precision
        valid_results['recall'] = valid_recall
        valid_results['fscore'] = valid_fscore

        test_precision, test_recall, test_fscore, _ = self._get_results(self.test_gold, self.test_predicted)
        test_results['precision'] = test_precision
        test_results['recall'] = test_recall
        test_results['fscore'] = test_fscore

        self._reset()

    def _get_results(self, gold, predicted):
        return precision_recall_fscore_support(gold, predicted, average=self.f1_average)
