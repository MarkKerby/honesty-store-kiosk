import _ from 'lodash';
import * as tf from '@tensorflow/tfjs';

const plotLoss = _.noop;
const plotAccuracy = _.noop;

/**
 * Compile and train the given model.
 *
 * @param {*} model The model to
 */
export default async function train(
  dataSource,
  model,
  onIteration = _.noop,
  onBatchComplete = _.noop
) {
  console.log('Training model...');

  // Now that we've defined our model, we will define our optimizer. The
  // optimizer will be used to optimize our model's weight values during
  // training so that we can decrease our training loss and increase our
  // classification accuracy.

  // The learning rate defines the magnitude by which we update our weights each
  // training step. The higher the value, the faster our loss values converge,
  // but also the more likely we are to overshoot optimal parameters
  // when making an update. A learning rate that is too low will take too long
  // to find optimal (or good enough) weight parameters while a learning rate
  // that is too high may overshoot optimal parameters. Learning rate is one of
  // the most important hyperparameters to set correctly. Finding the right
  // value takes practice and is often best found empirically by trying many
  // values.
  const LEARNING_RATE = 0.01;

  // We are using rmsprop as our optimizer.
  // An optimizer is an iterative method for minimizing an loss function.
  // It tries to find the minimum of our loss function with respect to the
  // model's weight parameters.
  const optimizer = 'rmsprop';

  // We compile our model by specifying an optimizer, a loss function, and a
  // list of metrics that we will use for model evaluation. Here we're using a
  // categorical crossentropy loss, the standard choice for a multi-class
  // classification problem like MNIST digits.
  // The categorical crossentropy loss is differentiable and hence makes
  // model training possible. But it is not amenable to easy interpretation
  // by a human. This is why we include a "metric", namely accuracy, which is
  // simply a measure of how many of the examples are classified correctly.
  // This metric is not differentiable and hence cannot be used as the loss
  // function of the model.
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  // Batch size is another important hyperparameter. It defines the number of
  // examples we group together, or batch, between updates to the model's
  // weights during training. A value that is too low will update weights using
  // too few examples and will not generalize well. Larger batch sizes require
  // more memory resources and aren't guaranteed to perform better.
  const batchSize = 320;

  // Leave out the last 15% of the training data for validation, to monitor
  // overfitting during training.
  const validationSplit = 0.15;

  const trainEpochs = 3;

  // We'll keep a buffer of loss and accuracy values over time.
  let trainBatchCount = 0;

  const trainData = dataSource.getTrainData();
  const testData = dataSource.getTestData();

  const totalNumBatches =
    Math.ceil((trainData.xs.shape[0] * (1 - validationSplit)) / batchSize) *
    trainEpochs;

  // During the long-running fit() call for model training, we include
  // callbacks, so that we can plot the loss and accuracy values in the page
  // as the training progresses.
  let valAcc;
  await model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;

        onBatchComplete(trainBatchCount, totalNumBatches);

        console.log(
          `Training... (${((trainBatchCount / totalNumBatches) * 100).toFixed(
            1
          )}% complete)`
        );

        plotLoss(trainBatchCount, logs.loss, 'train');
        plotAccuracy(trainBatchCount, logs.acc, 'train');
        if (batch % 10 === 0) {
          onIteration('onBatchEnd', batch, logs);
        }
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        valAcc = logs.val_acc;
        plotLoss(trainBatchCount, logs.val_loss, 'validation');
        plotAccuracy(trainBatchCount, logs.val_acc, 'validation');

        onIteration('onEpochEnd', epoch, logs);

        await tf.nextFrame();
      }
    }
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  console.log(
    `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
      `Final test accuracy: ${testAccPercent.toFixed(1)}%`
  );

  return {
    finalValAccPercent,
    testAccPercent
  };
}
