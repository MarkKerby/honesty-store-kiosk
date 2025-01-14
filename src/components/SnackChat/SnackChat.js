import React, {Component} from 'react';
import PropTypes from 'prop-types';
import {Stage, Provider} from '@inlet/react-pixi';

import FilterView from './FilterView';
import WebcamCapture from '../WebcamCapture/WebcamCapture';
import BackButton from '../BackButton/BackButton';
import GetBBox from '../GetBBox/GetBBox';

import * as posenet from '@tensorflow-models/posenet';
import './SnackChat.css';

const FEED_SIZE = 768;
const CAPTURE_SIZE = 500;
const COUNTDOWN_DURATION = 5;

const LOADING_CAPTIONS = [
  'Crunching bits',
  'Taking bytes',
  'Censoring inappropriate content',
  'Zooming... Enhancing',
  'Removing unsanctioned emotions'
];

function randomCaption() {
  const index = Math.floor(
    Math.random() * Math.random() * LOADING_CAPTIONS.length
  );
  return LOADING_CAPTIONS[index];
}

function normalise(n) {
  return (n *= FEED_SIZE / CAPTURE_SIZE);
}

class SnackChat extends Component {
  state = {
    loading: true,
    horizontalItem: false,
    countdown: COUNTDOWN_DURATION
  };

  webcamCap = React.createRef();
  filter = React.createRef();

  componentDidMount() {
    posenet.load(0.5).then(net => {
      this.net = net;
      this.setState({loading: false});
      this.ticker = setInterval(() => {
        this.setState(prevState => {
          if (prevState.countdown === 1) clearInterval(this.ticker);
          return {countdown: prevState.countdown - 1};
        });
      }, 1000);
    });
  }

  componentDidUpdate() {
    if (this.state.countdown === 0) {
      requestAnimationFrame(() => requestAnimationFrame(this.captureSnackChat));
    }
  }

  componentWillUnmount() {
    clearInterval(this.ticker);
  }

  getPose = async () => {
    if (!this.net || !this.webcamCap.current) return null;
    this.canvas = this.webcamCap.current.getCanvas();

    const pose = await this.net.estimateSinglePose(this.canvas, 0.3, false, 16);
    const body = {
      ears: this.calcAngles({
        leftX: normalise(pose.keypoints[4].position.x),
        leftY: normalise(pose.keypoints[4].position.y),
        rightX: normalise(pose.keypoints[3].position.x),
        rightY: normalise(pose.keypoints[3].position.y)
      }),
      shoulders: this.calcAngles({
        leftX: normalise(pose.keypoints[6].position.x),
        leftY: normalise(pose.keypoints[6].position.y),
        rightX: normalise(pose.keypoints[5].position.x),
        rightY: normalise(pose.keypoints[5].position.y)
      })
    };

    return body;
  };

  calcAngles = bodyPart => {
    bodyPart.width = Math.abs(bodyPart.rightX - bodyPart.leftX);
    bodyPart.height = bodyPart.rightY - bodyPart.leftY;
    bodyPart.span = Math.sqrt(bodyPart.width ** 2 + bodyPart.height ** 2);
    bodyPart.angle = Math.atan(bodyPart.height / bodyPart.width);
    bodyPart.angle += this.state.horizontalItem ? Math.PI / 2 : 0;
    return bodyPart;
  };

  onFail = () => {
    this.props.setSendWithPhoto(false);
    this.props.history.replace('/slackname');
  };

  onBack = () => {
    this.backClicked = true;
    clearInterval(this.timer);
    this.props.history.replace(
      this.props.actualItem === this.props.predictionID
        ? '/confirmitem'
        : '/editsnack'
    );
  };

  setHorizontalFlag = bboxes => {
    if (bboxes[0].height < bboxes[0].width) {
      this.setState({horizontalItem: true});
    }
  };

  captureSnackChat = async () => {
    const filter = await this.filter.current.toImage();
    window.filter = filter;

    const canvas = this.canvas;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(filter, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE);
    this.props.setSnackChat(canvas.toDataURL());
    this.props.history.replace('/slackname');
  };

  render() {
    return (
      <div className="page">
        <div
          id="overlay"
          className={this.state.countdown <= 0 ? 'flash' : ''}
        />
        <header className="header">
          <BackButton handleClick={this.onBack} />
          <div className="header-text">
            {this.state.countdown > 0
              ? `Taking photo in ${this.state.countdown}...`
              : `${randomCaption()}...`}
          </div>
        </header>
        <div>
          <WebcamCapture
            imgSize={CAPTURE_SIZE}
            onFail={this.onFail}
            ref={this.webcamCap}
          />
          {!this.state.loading &&
            this.webcamCap.current.webcam.current && (
              <Stage
                width={FEED_SIZE}
                height={FEED_SIZE}
                options={{transparent: true}}
                className="snackchat-stage"
                style={{
                  visibility: FilterView.LIVE_PREVIEW ? 'visible' : 'hidden'
                }}>
                <Provider>
                  {app => (
                    <FilterView
                      image={this.props.storeList[this.props.actualItem].image}
                      app={app}
                      getPose={this.getPose}
                      video={this.webcamCap.current.webcam.current.video}
                      ref={this.filter}
                    />
                  )}
                </Provider>
              </Stage>
            )}
        </div>
        <GetBBox
          svg={this.props.storeList[this.props.actualItem].image}
          callback={this.setHorizontalFlag}
        />
      </div>
    );
  }
}

SnackChat.propTypes = {
  setSnackChat: PropTypes.func.isRequired,
  setSendWithPhoto: PropTypes.func.isRequired,
  history: PropTypes.object.isRequired,
  storeList: PropTypes.object.isRequired,
  actualItem: PropTypes.string.isRequired,
  predictionID: PropTypes.string
};

export default SnackChat;
