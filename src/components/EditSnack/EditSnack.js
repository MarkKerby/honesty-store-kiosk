import React, {Component} from 'react';
import PropTypes from 'prop-types';

import BackButton from '../BackButton/BackButton';
import ConfirmationModal from '../ConfirmationModal/ConfirmationModal';
import ListSelection from '../listSelection/ListSelection';

import './EditSnack.css';

class EditSnack extends Component {
  state = {
    selection: null
  };

  goBack = async () => {
    const lastPage = this.props.prediction.id ? 'confirmitem' : 'scanitem';
    this.props.history.replace(lastPage);
  };

  handleClick = () => {
    this.props.setActualItem(this.state.selection.id);
    const nextPage = this.props.sendWithPhoto ? 'snackchat' : 'slackname';
    this.props.history.replace('/' + nextPage);
  };

  promptToConfirm = selection => {
    this.setState({selection});
  };

  deselect = () => {
    this.setState(
      prevState => (prevState.selection ? {selection: null} : null)
    );
  };

  render() {
    return (
      <div className="edit-snack--page">
        <header className="header">
          <BackButton handleClick={this.goBack} className="massiveZIndex" />
          <div className="header-text">
            Sorry, I can’t recognise that snack
            <br />
            Please select it below
          </div>
        </header>
        <ListSelection
          items={this.props.items}
          onClick={this.promptToConfirm}
          additionalOptions={{
            heading: 'Suggestions',
            options: this.props.suggestions
          }}
          selected={this.state.selection ? this.state.selection.name : null}
        />
        {this.state.selection && (
          <ConfirmationModal
            disabled={this.state.sending}
            onClick={this.handleClick}
          />
        )}
      </div>
    );
  }
}

EditSnack.propTypes = {
  setActualItem: PropTypes.func.isRequired,
  sendWithPhoto: PropTypes.bool.isRequired,
  history: PropTypes.object.isRequired,
  items: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      id: PropTypes.string.isRequired
    }).isRequired
  ).isRequired,
  suggestions: PropTypes.arrayOf(PropTypes.object).isRequired,
  prediction: PropTypes.object,
  dataController: PropTypes.object.isRequired
};

export default EditSnack;
