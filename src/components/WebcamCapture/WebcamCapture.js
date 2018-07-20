import React, {Component} from 'react';
import Webcam from 'react-webcam';
import Notification from './../Notification/Notification';
import PropTypes from 'prop-types';
import './WebcamCapture.css';

const height = 768;
const width = 768;

class WebcamCapture extends Component {
	state = {
		isDetecting: true,
		cameraConnected: false
	};

	webcam = React.createRef();

	setupScreenshotInterval() {
		this.ticker = setInterval(() => {
			const img = new Image(224, 224);
			img.src = this.webcam.current.getScreenshot();
			img.onload = () => {
				this.props.onImgLoad(img);
			};
		}, this.props.interval);
	}

	setupWebcam() {
		navigator.mediaDevices
			.getUserMedia({video: true})
			.then(() => {
				this.setState({
					cameraConnected: true,
					isDetecting: false
				});
				this.setupScreenshotInterval();
			})
			.catch(() =>
				this.setState({cameraConnected: false, isDetecting: false})
			);
	}

	componentDidMount() {
		if (!navigator.mediaDevices) return;
		this.setupWebcam();
	}

	componentWillUnmount() {
		clearInterval(this.ticker);
	}

	render() {
		if (this.state.isDetecting) {
			return null;
		}

		if (this.state.cameraConnected) {
			return (
				<div className="webcam-container">
					<Webcam
						audio={false}
						height={height}
						ref={this.webcam}
						screenshotFormat="image/jpeg"
						width={width}
						className="webcam-capture--video"
						screenshotWidth={300}
					/>
				</div>
			);
		}
		return (
			<Notification message="failed to load video feed" isError={true} />
		);
	}
}

WebcamCapture.propTypes = {
	onImgLoad: PropTypes.func.isRequired,
	interval: PropTypes.number.isRequired
};

export default WebcamCapture;
