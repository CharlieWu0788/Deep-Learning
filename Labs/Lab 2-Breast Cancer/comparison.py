# Part 2: Model comparison (see README.md).
#
# A note on using AI for this file:
# It is totally OK to use an LLM (ChatGPT, Codex, Claude Code, etc.) to write
# this code. But before you ask, you must give it context: paste
# test_comparison.py and binary_classification.py into the chat, or open this
# folder in Codex / Claude Code so it can read them. Without that, it cannot
# know what load_data() returns or what the tests check.
#
# You also have to pick your scikit-learn model first (see the README for
# options). The model choice is yours, and it is what drives the code
# generation.
#
# The course AI policy still applies: include your transcript and be ready to
# explain the code verbally.
"""
Part 2: Model Comparison

This comparison uses a linear Support Vector Machine (SVM) from scikit-learn.
It fits a separating hyperplane by balancing a wide margin against penalties
for margin violations, with C=1.0 controlling that tradeoff.
The from-scratch sigmoid classifier instead minimizes half-squared error.
A linear SVM provides a useful comparison because both models learn linear
decision boundaries but optimize different objectives on the same data.
"""

from sklearn.svm import SVC

from binary_classification import load_data, train, predict, accuracy


# Load the same normalized breast cancer dataset used by the from-scratch model.
X_train, X_test, y_train, y_test, feature_names = load_data()

# Train the from-scratch sigmoid classifier using the same settings as Part 1.
w, b, losses = train(
    X_train,
    y_train,
    alpha=0.01,
    n_epochs=100,
    verbose=False,
)

# Evaluate the from-scratch classifier.
scratch_train_pred = predict(X_train, w, b)
scratch_test_pred = predict(X_test, w, b)

scratch_train_accuracy = accuracy(y_train, scratch_train_pred)
scratch_test_accuracy = accuracy(y_test, scratch_test_pred)

# Train a linear SVM on the same normalized training data.
svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train.numpy(), y_train.numpy())

# Evaluate the SVM on both training and test data.
svm_train_pred = svm.predict(X_train.numpy())
svm_test_pred = svm.predict(X_test.numpy())

svm_train_accuracy = (svm_train_pred == y_train.numpy()).mean()
svm_test_accuracy = (svm_test_pred == y_test.numpy()).mean()


print("Model Comparison")
print("=" * 50)
print(f"From-scratch classifier - Train accuracy: {scratch_train_accuracy:.4f}")
print(f"From-scratch classifier - Test accuracy:  {scratch_test_accuracy:.4f}")
print(f"SVM                     - Train accuracy: {svm_train_accuracy:.4f}")
print(f"SVM                     - Test accuracy:  {svm_test_accuracy:.4f}")

# On the fixed random_state=42 split, the from-scratch model achieved 99.12%
# test accuracy versus 95.61% for the linear SVM, despite both reaching 98.68%
# training accuracy.
# Their different objectives and the SVM's default regularization can produce
# different boundaries, which may explain the four-test-sample advantage here.
# This single split does not establish that the from-scratch model is generally
# better; repeated splits or cross-validation would be needed to assess that.

print()
print(
    "The from-scratch classifier and the linear SVM both learn linear "
    "decision boundaries, but they use different optimization objectives. "
    "Their accuracy results show how the choice of training objective can "
    "affect performance on the same breast cancer dataset."
)
