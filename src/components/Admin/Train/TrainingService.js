import * as tf from '@tensorflow/tfjs';
import _ from 'lodash';
import createConvModel from './ModelFactory';
import {DataService} from './DataService';

export default class TrainingService {
  model = null;
  dataService = new DataService();

  makeNewModel() {
    this.model = createConvModel();

    this.model.compile({
      optimizer: 'rmsprop',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }

  async fetchTrainingDataDetails() {
    return this.dataService.prepare();
  }

  log(message) {
    console.log(message);
  }

  async go(progressCallback = _.noop) {
    this.log('Beginning training process');

    this.makeNewModel();
    this.log('Created a new model');

    await this.fetchTrainingDataDetails();
    this.log('Fetched training data details');

    const totalTrainingImages = this.dataService.getTotalTrainingImages();
    const imagesPerBatch = 3;
    let imagesProcessed = 0;
    let batchNumber = 0;

    while (imagesProcessed < totalTrainingImages) {
      this.log(`Requesting batch ${batchNumber} of size ${imagesPerBatch}`);
      const nextBatch = await this.dataService.getNextTrainingBatchOf(
        imagesPerBatch,
        batchNumber
      );
      console.log(nextBatch);
      const totalInBatch = nextBatch.length;

      this.log(`Batch: ${batchNumber} contains ${totalInBatch} samples`);
      /*
      
      const imageTensors = nextBatch.map(batch => batch.imageTensor);
      const labelTensors = nextBatch.map(batch => batch.labelTensor);

      console.log(imageTensors);
      console.log(labelTensors);
*/
      await Promise.all(
        nextBatch.map(async (batch, n) => {
          return this.model.trainOnBatch(batch.imageTensor, batch.labelTensor);
        })
      );
      /*
      await this.model.fit(
        imageTensors,
        labelTensors,
        {
          batchSize: totalInBatch,
          validationSplit: 0.0,
          epochs: 3,
          callbacks: {
            onBatchEnd: async (batch, logs) => {
              this.log("A batch has ended")
              await tf.nextFrame();
            },
            onEpochEnd: async (epoch, logs) => {
              this.log("An epoch has ended")
              await tf.nextFrame();
            },
          }
        }
      );
*/
      this.log(`Applied fit for batch ${batchNumber}`);
      imagesProcessed = imagesProcessed + totalInBatch;
      ++batchNumber;
    }

    this.log('Training complete');
  }
}
