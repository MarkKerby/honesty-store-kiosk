import _ from 'lodash';
import * as tf from '@tensorflow/tfjs';
import DataController from '../utils/DataController';

const DATABASE_BATCH_SIZE = 100;
const MAX_DATABASE_BATCHES = 3;

const PERCENT_TO_TEST = 0.15;

const IMAGE_W = 224;
const IMAGE_H = 224;

export class DataService {
  dataController = new DataController();

  log(message) {
    console.log(message);
  }

  async getNextTrustedImages(lastDocumentTimestamp) {
    return this.dataController.getImages(
      true,
      DATABASE_BATCH_SIZE,
      lastDocumentTimestamp // Optional, if provided results will begin after this previous result
    );
  }

  async fetchAllTrustedImagesMetaData() {
    const allTrustedImages = [];

    let batchNumber = 0;
    let lastDocumentTimestamp = undefined;

    do {
      this.log('Reading batch ' + batchNumber);
      const nextBatch = await this.getNextTrustedImages(lastDocumentTimestamp);

      lastDocumentTimestamp = Math.max(...nextBatch.map(doc => doc.timestamp));
      allTrustedImages.push(...nextBatch);

      this.log('Batch ' + batchNumber + ' has ' + nextBatch.length + ' items');

      if (nextBatch.length !== DATABASE_BATCH_SIZE) break;
    } while (batchNumber++ < MAX_DATABASE_BATCHES);

    return allTrustedImages;
  }

  async prepare() {
    this.log('Loading honesty store shop info');
    const store = await this.dataController.getStoreList();
    this.log('Have shop info');

    this.log('Fetching all trusted image meta data');
    const allTrustedImages = _.shuffle(
      await this.fetchAllTrustedImagesMetaData()
    );
    this.log(
      `Have image meta data, there are ${
        allTrustedImages.length
      } trusted images available`
    );

    const splitAt = Math.floor(allTrustedImages.length * (1 - PERCENT_TO_TEST));
    this.trainingSource = allTrustedImages.slice(0, splitAt);
    this.testingSource = allTrustedImages.slice(splitAt);
    this.log(
      `Will use ${this.trainingSource.length} for training and ${
        this.testingSource.length
      } for testing`
    );
  }

  loadCanvasFromUrl(url) {
    return new Promise(resolve => {
      const img = new Image();

      img.crossOrigin = '';
      img.src = url;
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = IMAGE_W;
        canvas.height = IMAGE_H;

        canvas
          .getContext('2d')
          .drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas);
      };
    });
  }

  async getBatchOf(size, step, sourceBuffer) {
    const startIndex = size * step;
    const source = sourceBuffer.slice(startIndex, startIndex + size);

    const imagePromises = source.map(s => this.loadCanvasFromUrl(s.url));

    const imageTensors = await Promise.all(imagePromises).then(
      this.canvasesToImageTensors.bind(this)
    );

    const labelTensors = source.map(sourceItem => {
      const charArray = Uint8Array.from(
        sourceItem.label.split('').map(x => x.charCodeAt(0))
      );

      return tf.tensor2d(charArray, [1, 36]);
    });

    return _.zipWith(imageTensors, labelTensors, (imageTensor, labelTensor) => {
      return {
        imageTensor,
        labelTensor
      };
    });
  }

  getTotalTrainingImages() {
    return this.trainingSource.length;
  }

  getTotalTestingImages() {
    return this.testingSource.length;
  }

  async getNextTrainingBatchOf(size, step) {
    return this.getBatchOf(size, step, this.trainingSource);
  }

  async getNextTestingBatchOf(size, step) {
    return this.getBatchOf(size, step, this.testingSource);
  }

  imageToTensor = image => {
    return tf.tidy(() =>
      tf
        .fromPixels(image)
        .expandDims(0)
        .toFloat()
        .div(tf.scalar(127))
        .sub(tf.scalar(1))
    );
  };

  canvasesToImageTensors(canvases) {
    return canvases.map(this.imageToTensor);
  }
}
