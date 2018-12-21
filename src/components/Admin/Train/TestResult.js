import React, {Component} from 'react';
import PropTypes from 'prop-types';
import * as tf from '@tensorflow/tfjs';

import './TestResult.css';

class TestResult extends Component {
  componentWillUpdate() {
    this.draw();
  }

  render() {
    const correct = this.props.prediction === this.props.label;
    return (
      <div className={`test-result ${correct ? 'correct' : 'incorrect'}`}>
        <span>{this.props.prediction}</span>
        <canvas ref="canvas" />
      </div>
    );
  }

  draw() {
    tf.tidy(() => {
      const imageData = this.props.imageData;
      const canvas = this.refs.canvas;

      const [width, height] = [28, 28];
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.putImageData(imageData, 0, 0);
    });
  }
}

TestResult.propTypes = {
  prediction: PropTypes.string.isRequired,
  label: PropTypes.string.isRequired,
  imageData: PropTypes.object.isRequired
};

export default TestResult;
