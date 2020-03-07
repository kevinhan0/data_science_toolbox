import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from functools import partial
from itertools import product
from typing import Any, Union
import matplotlib.pylab as plt
from dataclasses import dataclass
from ipywidgets import interactive
from IPython.display import display
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def test_validate_arguments():

    try:
        _validate_arguments(X_train=1)
    except ValueError as e:
        logging.info("Error catched: %s" % e)

    try:
        _validate_arguments(X_train=1, X_test=1)
    except ValueError as e:
        logging.info("Error catched: %s" % e)

    try:
        _validate_arguments(y=1)
    except ValueError as e:
        logging.info("Error catched: %s" % e)

    try:
        _validate_arguments(X_train=1, y_train=1, X=1, y=1)
    except ValueError as e:
        logging.info("Error catched: %s" % e)

    train_only, _, _, _ = _validate_arguments(X_train=1, y_train=1)
    if train_only:
        logging.info("SUCCESS: Training set only.")
    else:
        logging.warning("FAILURE: Training set only.")

    _, test_only, _, _ = _validate_arguments(X_test=1, y_test=1)
    if test_only:
        logging.info("SUCCESS: Test set only.")
    else:
        logging.warning("FAILURE: Test set only.")

    _, _, both_not_cv, _ = _validate_arguments(X_train=1, y_train=1, X_test=1, y_test=1)
    if both_not_cv:
        logging.info("SUCCESS: Both training and test sets.")
    else:
        logging.warning("FAILURE: Both training and test sets.")

    _, _, _, cv_only = _validate_arguments(X=1, y=1)
    if cv_only:
        logging.info("SUCCESS: CV only.")
    else:
        logging.warning("FAILURE: CV only.")


@dataclass
class ModelReport:

    estimator: Any
    X_train: Union[pd.DataFrame, None] = None
    y_train: Union[pd.DataFrame, None] = None
    X_test: Union[pd.DataFrame, None] = None
    y_test: Union[pd.DataFrame, None] = None
    X: Union[pd.DataFrame, None] = None
    y: Union[pd.DataFrame, None] = None
    cv: Union[int, Any] = 5

    def __post_init__(self):

        # Get estimator name
        self.estimator_name = type(self.estimator).__name__

        # Check if estimator has predict_proba method
        self._has_proba = hasattr(self.estimator, "predict_proba")

        # Check argument validity
        train_only, test_only, both_not_cv, cv_only = self._validate_arguments()

        # Generate dropdown menu options
        self.options = self._generate_dropdown_options(
            train_only, test_only, both_not_cv, cv_only
        )

        # Extract feature names
        self.feature_names = self._extract_feature_names(
            train_only, test_only, both_not_cv, cv_only
        )

        # Build dashboard layout
        self.tabs = self._build_dashboard_layout()

    def _validate_arguments(self):

        # Check data
        train_cond = self.X_train is None or self.y_train is None
        test_cond = self.X_test is None or self.y_test is None
        cv_cond = self.X is None or self.y is None

        # Check argument sanity
        if (train_cond and test_cond) and cv_cond:
            raise ValueError(
                "Must pass in either one of training set, test set, or the original dataset before train-test split."
            )
        elif not cv_cond and (not train_cond or not test_cond):
            raise ValueError(
                "Pass in either training and/or test set or the original dataset before train-test split."
            )

        # Determine which data is estimator being evaluated on
        train_only = not train_cond and test_cond
        test_only = train_cond and not test_cond
        both_not_cv = not train_cond and not test_cond
        cv_only = train_cond and test_cond and not cv_cond

        return train_only, test_only, both_not_cv, cv_only

    @staticmethod
    def _generate_dropdown_options(train_only, test_only, both_not_cv, cv_only):

        # Selector options
        if train_only:
            options = ["Train"]
        elif test_only:
            options = ["Validation"]
        elif both_not_cv:
            options = ["(All)", "Train", "Validation"]
        else:
            options = ["CV Results"]

        return options

    def _extract_feature_names(self, train_only, test_only, both_not_cv, cv_only):

        # Selector options
        if train_only:
            feature_names = (
                self.X_train.columns.tolist()
                if isinstance(self.X_train, pd.DataFrame)
                else np.arange(0, self.X_train.shape[1])
            )
        elif test_only:
            feature_names = (
                self.X_test.columns.tolist()
                if isinstance(self.X_test, pd.DataFrame)
                else np.arange(0, self.X_test.shape[1])
            )
        elif both_not_cv:
            feature_names = (
                self.X_train.columns.tolist()
                if isinstance(self.X_train, pd.DataFrame)
                else np.arange(0, self.X_train.shape[1])
            )
        else:
            feature_names = (
                self.X.columns.tolist()
                if isinstance(self.X, pd.DataFrame)
                else np.arange(0, self.X.shape[1])
            )

        return feature_names

    def _build_dashboard_layout(self):

        # ROC tab
        self._roc_output = widgets.Output()
        self._roc_selector = widgets.Dropdown(
            options=self.options,
            value=self.options[0],
            description="Option: ",
            continuous_update=True,
        )
        roc_tab = widgets.VBox(children=[self._roc_selector, self._roc_output])

        # Confusion matrix tab
        self._cm_floatslider = widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, step=0.05, description="Threshold: ",
        )
        self._cm_selector = widgets.Dropdown(
            options=self.options, value=self.options[0], description="Option: "
        )
        cm_hbar = widgets.HBox(
            children=[self._cm_selector, self._cm_floatslider]
        )
        self._cm_output = widgets.Output()
        cm_tab = widgets.VBox(children=[cm_hbar, self._cm_output])

        # Precision-recall curve tab
        self._prc_output = widgets.Output()
        self._prc_selector = widgets.Dropdown(
            options=self.options,
            value=self.options[0],
            description="Option: ",
            continuous_update=True,
        )
        prc_tab = widgets.VBox(children=[self._prc_selector, self._prc_output])

        # Feature importance tab
        tabs_children = [roc_tab, cm_tab, prc_tab]
        if self.feature_importance is not None:
            self._feature_importance_output = widgets.Output()
            feature_importance_tab = widgets.VBox(
                children=[self._feature_importance_output]
            )
            tabs_children.append(feature_importance_tab)

        # Build dashboard layout
        tabs = widgets.Tab(children=tabs_children)
        tabs.set_title(0, "ROC Curve")
        tabs.set_title(1, "Confusion Matrix")
        tabs.set_title(2, "Precision-Recall Curve")
        if self.feature_importance is not None:
            tabs.set_title(3, "Feature Importance")

        return tabs

    @property
    def y_train_score(self):
        return (
            self.estimator.predict_proba(self.X_train)[:,1]
            if self.X_train is not None and self._has_proba
            else self.X_train
        )

    @property
    def y_test_score(self):
        return (
            self.estimator.predict_proba(self.X_test)[:,1]
            if self.X_test is not None and self._has_proba
            else self.X_test
        )

    @property
    def feature_importance(self):
        if hasattr(self.estimator, "feature_importances_"):
            return (
                pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": self.estimator.feature_importances_,
                    }
                )
                .sort_values(by="importance", ascending=False)
                .reset_index(drop=True)
            )
        else:
            return None

    def _plot_feature_importance(self):
        _, ax = plt.subplots(figsize=(10, min(len(self.feature_importance), 14)))
        bar = (
            self.feature_importance.sort_values(by="importance")
            .head(50)
            .set_index("feature")
            .plot.barh(ax=ax, color="white", edgecolor="black")
        )
        for b in bar.patches:
            b.set_hatch("//")
        ax.set_title("Feature Importance")
        ax.get_legend().remove()
        plt.show()

    @staticmethod
    def plot_roc_curve(
        option,
        y_train,
        y_train_score,
        y_test,
        y_test_score,
        estimator_name,
        roc_output,
    ):
        def _plotter(y_true, y_score, ax, label_prefix):
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            label = "%s (AUC = %.2f)" % (label_prefix, auc)

            ax.plot(fpr, tpr, label=label)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic Curve")
            ax.legend(loc="lower right")

        roc_output.clear_output()

        if option == "(All)":
            with roc_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_train,
                    y_train_score,
                    ax=ax,
                    label_prefix="Training: %s" % estimator_name,
                )
                _plotter(
                    y_test,
                    y_test_score,
                    ax=ax,
                    label_prefix="Validation: %s" % estimator_name,
                )
                plt.show()
        elif option == "Train":
            with roc_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_train,
                    y_train_score,
                    ax=ax,
                    label_prefix="Training: %s" % estimator_name,
                )
                plt.show()
        elif option == "Validation":
            with roc_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_test,
                    y_test_score,
                    ax=ax,
                    label_prefix="Validation: %s" % estimator_name,
                )
                plt.show()

    @staticmethod
    def plot_confusion_matrix(
        threshold,
        option,
        y_train,
        y_train_score,
        y_test,
        y_test_score,
        estimator,
        cm_output,
        normalize="all",
    ):
        def _plotter(y_true, y_score, threshold, ax=None, title=""):

            # Get confusion matrix on threshold
            y_pred = [int(pred >= threshold) for pred in y_score]
            cm = confusion_matrix(y_true, y_pred, normalize=normalize)

            # Get estimator classes
            estimator_classes = (
                estimator.classes_
                if hasattr(estimator, "classes_")
                else np.arange(0, cm.shape[0])
            )

            # Initialize figure
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            # Build colorscale
            n_classes = cm.shape[0]
            im_ = ax.imshow(cm, interpolation="nearest", cmap="binary")
            cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

            # Print text with appropriate color depending on background
            text_ = np.empty_like(cm, dtype=object)
            values_format = ".2g"
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                text_[i, j] = ax.text(
                    j,
                    i,
                    format(cm[i, j], values_format),
                    ha="center",
                    va="center",
                    color=color,
                )

            # Build figure labels
            fig.colorbar(im_, ax=ax)
            ax.set(
                xticks=np.arange(n_classes),
                yticks=np.arange(n_classes),
                xticklabels=estimator_classes,
                yticklabels=estimator_classes,
                ylabel="True label",
                xlabel="Predicted label",
                title=title,
            )
            ax.set_ylim((n_classes - 0.5, -0.5))
            plt.setp(ax.get_xticklabels(), rotation="horizontal")

        cm_output.clear_output()

        if option == "(All)":
            with cm_output:
                f, (ax1, ax2) = plt.subplots(2, figsize=(10, 16))
                _plotter(
                    y_train,
                    y_train_score,
                    threshold,
                    ax=ax1,
                    title="Training: Confusion Matrix @ Threshold = %.2f" % threshold,
                )
                _plotter(
                    y_test,
                    y_test_score,
                    threshold,
                    ax=ax2,
                    title="Validation: Confusion Matrix @ Threshold = %.2f" % threshold,
                )
                plt.show()
        elif option == "Train":
            with cm_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_train,
                    y_train_score,
                    threshold,
                    ax=ax,
                    title="Training: Confusion Matrix @ Threshold = %.2f" % threshold,
                )
                plt.show()
        elif option == "Validation":
            with cm_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_test,
                    y_test_score,
                    threshold,
                    ax=ax,
                    title="Validation: Confusion Matrix @ Threshold = %.2f" % threshold,
                )
                plt.show()

    @staticmethod
    def plot_precision_recall_curve(
        option,
        y_train,
        y_train_score,
        y_test,
        y_test_score,
        estimator_name,
        prc_output,
    ):
        def _plotter(y_true, y_score, ax=None, label_prefix=""):

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            average_precision = average_precision_score(y_true, y_score)

            if ax is None:
                fig, ax = plt.subplots()

            line_kwargs = {
                "label": "%s (AP = %.2f)" % (label_prefix, average_precision),
                "drawstyle": "steps-post"
            }

            line_, = ax.plot(recall, precision, **line_kwargs)
            ax.set(xlabel="Recall", ylabel="Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc='lower left')

        prc_output.clear_output()

        if option == "(All)":
            with prc_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_train,
                    y_train_score,
                    ax=ax,
                    label_prefix="Training: %s" % estimator_name,
                )
                _plotter(
                    y_test,
                    y_test_score,
                    ax=ax,
                    label_prefix="Validation: %s" % estimator_name,
                )
                plt.show()
        elif option == "Train":
            with prc_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_train,
                    y_train_score,
                    ax=ax,
                    label_prefix="Training: %s" % estimator_name,
                )
                plt.show()
        elif option == "Validation":
            with prc_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_test,
                    y_test_score,
                    ax=ax,
                    label_prefix="Validation: %s" % estimator_name,
                )
                plt.show()

    @staticmethod
    def plot_classification_report(
        threshold,
        option,
        y_train,
        y_train_score,
        y_test,
        y_test_score,
        estimator,
        cr_output,
    ):
        def _plotter(X, y_score, threshold, ax=None):
            warnings.filterwarnings("ignore")

            # Get confusion matrix on threshold
            y_pred = [int(pred >= threshold) for pred in y_score]

            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            viz = ClassificationReport(
                estimator,
                cmap="Greys",
                support="percent",
                is_fitted=True,
                ax=ax
            )
            viz.score(X, y_pred)
            viz.show()
 
        cr_output.clear_output()

        if option == "(All)":
            with cr_output:
                f, (ax1, ax2) = plt.subplots(2, figsize=(10, 16))
                _plotter(
                    y_train,
                    y_train_score,
                    threshold,
                    ax=ax1,
                    title="Training: Classification Report @ Threshold = %.2f" % threshold,
                )
                _plotter(
                    y_test,
                    y_test_score,
                    threshold,
                    ax=ax2,
                    title="Validation: Classification Report @ Threshold = %.2f" % threshold,
                )
                plt.show()
        elif option == "Train":
            with cr_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_train,
                    y_train_score,
                    threshold,
                    ax=ax,
                    title="Training: Classification Report @ Threshold = %.2f" % threshold,
                )
                plt.show()
        elif option == "Validation":
            with cr_output:
                f, ax = plt.subplots(figsize=(10, 8))
                _plotter(
                    y_test,
                    y_test_score,
                    threshold,
                    ax=ax,
                    title="Validation: Classification Report @ Threshold = %.2f" % threshold,
                )
                plt.show()

    def evaluate(self):

        # Feature importance
        if self.feature_importance is not None:
            with self._feature_importance_output:
                self._plot_feature_importance()

        # ROC curve
        prc = partial(
            ModelReport.plot_roc_curve,
            y_train=self.y_train,
            y_train_score=self.y_train_score,
            y_test=self.y_test,
            y_test_score=self.y_test_score,
            estimator_name=self.estimator_name,
            roc_output=self._roc_output,
        )
        with self._roc_output:
            prc(self.options[0])

        # Confusion matrix
        pcm = partial(
            ModelReport.plot_confusion_matrix,
            y_train=self.y_train,
            y_train_score=self.y_train_score,
            y_test=self.y_test,
            y_test_score=self.y_test_score,
            estimator=self.estimator,
            cm_output=self._cm_output,
        )
        with self._cm_output:
            pcm(option=self.options[0], threshold=0.5)

        # Precision-recall curve
        pprc = partial(
            ModelReport.plot_precision_recall_curve, 
            y_train=self.y_train,
            y_train_score=self.y_train_score,
            y_test=self.y_test,
            y_test_score=self.y_test_score,
            estimator_name=self.estimator_name,
            prc_output=self._prc_output,
        )
        with self._prc_output:
            pprc(self.options[0])

        # Event handlers
        def _eventhandler_plot_roc_curve(option):
            prc(option.new)

        def _eventhandler_plot_confusion_matrix_option(option):
            pcm(option=option.new, threshold=self._cm_floatslider.value)

        def _eventhandler_plot_confusion_matrix_threshold(threshold):
            pcm(option=self._cm_selector.value, threshold=threshold.new)

        def _eventhandler_plot_precision_recall_curve(option):
            pprc(option.new)

        self._roc_selector.observe(_eventhandler_plot_roc_curve, names="value")
        self._cm_floatslider.observe(_eventhandler_plot_confusion_matrix_threshold, names="value")
        self._cm_selector.observe(_eventhandler_plot_confusion_matrix_option, names="value")
        self._prc_selector.observe(_eventhandler_plot_precision_recall_curve, names="value")
        display(self.tabs)
