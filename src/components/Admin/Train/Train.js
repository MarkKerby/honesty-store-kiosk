import React, {Component} from 'react';
import {DataSource} from './DataSource';
import TestResult from './TestResult';
import * as tf from '@tensorflow/tfjs';
import './Train.css';
import createConvModel from './ModelFactory';
import train from './Trainer';

class Train extends Component {
  state = {
    working: false,
    got: false,
    examples: [],
    labels: [],
    predictions: [],

    trainBatchCount: 0,
    totalNumBatches: 0,
    finalValAccPercent: 0,
    testAccPercent: 0
  };

  model = null;

  render() {
    return (
      <div>
        <div>
          <button className="button button-admin" onClick={this.back}>
            &laquo; Back
          </button>
        </div>

        <h1>This is the training screen</h1>
        <button
          className="button button-admin"
          onClick={this.onGoClicked}
          disabled={this.state.working}>
          Train
        </button>

        {this.state.working && this.state.totalNumBatches > 0 && (
          <div>
            <div>Training…</div>
            <div>{`${(
              (this.state.trainBatchCount / this.state.totalNumBatches) *
              100
            ).toFixed(1)}%  complete`}</div>
            <div>To stop training, refresh or close page.</div>
          </div>
        )}

        {this.state.got && (
          <div>
            <div>{`Final validation accuracy: ${this.state.finalValAccPercent.toFixed(
              1
            )}%; `}</div>
            <div>{`Final test accuracy: ${this.state.testAccPercent.toFixed(
              1
            )}%`}</div>
          </div>
        )}

        <div className="box">
          {this.state.examples.map((exampleImage, n) => {
            return (
              <TestResult
                key={n}
                imageData={exampleImage}
                label={this.state.labels[n]}
                prediction={this.state.predictions[n]}
              />
            );
          })}
        </div>
      </div>
    );
  }

  back = () => {
    this.props.history.replace('/admin');
  };

  onGoClicked = event => {
    this.setState({
      working: true,
      got: false,

      trainBatchCount: 0,
      totalNumBatches: 0,
      finalValAccPercent: 0,
      testAccPercent: 0
    });

    this.doTrain();
  };

  doTrain() {
    this.loadImages()
      .then(() => this.createModel())
      .then(() => this.trainModel())
      .then(() => this.showPredictions())
      .then(() => {
        this.setState({
          working: false,
          got: true
        });
      });
  }

  loadImages() {
    this.dataSource = new DataSource();
    return this.dataSource.load();
  }

  showPredictions() {
    const examples = this.dataSource.getTestData(100);

    tf.tidy(() => {
      this.showTestResults(examples);
    });
  }

  showTestResults(batch) {
    const output = this.model.predict(batch.xs);
    const testExamples = batch.xs.shape[0];

    const axis = 1;
    const labels = Array.from(batch.labels.argMax(axis).dataSync()).map(
      l => `${l}`
    );
    const predictions = Array.from(output.argMax(axis).dataSync()).map(
      l => `${l}`
    );
    const examples = [];

    for (let i = 0; i < testExamples; i++) {
      const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

      const [width, height] = [28, 28];
      const imageData = new ImageData(width, height);
      const data = image.dataSync();
      for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        const value = data[i] * 255;
        imageData.data[j + 0] = value;
        imageData.data[j + 1] = value;
        imageData.data[j + 2] = value;
        imageData.data[j + 3] = 255;
      }

      examples.push(imageData);
    }

    this.setState({
      examples,
      labels,
      predictions
    });
  }

  createModel() {
    this.model = createConvModel();
    this.model.summary();
  }

  async trainModel() {
    const trained = await train(
      this.dataSource,
      this.model,
      undefined,
      (trainBatchCount, totalNumBatches) => {
        this.setState({
          trainBatchCount,
          totalNumBatches
        });
      }
    );

    this.setState({
      finalValAccPercent: trained.finalValAccPercent,
      testAccPercent: trained.testAccPercent
    });
  }
}

export default Train;
