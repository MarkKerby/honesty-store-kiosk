import React from 'react';
import './SuccessPage.css';
import Logo from './../Logo/Logo';
import Hand from './../Hand/Hand';
import TimeoutNotification from '../TimeoutNotification/TimeoutNotification';
import PropTypes from 'prop-types';

const SuccessPage = props => {
  return (
    <div onClick={() => props.history.push('/')}>
      <Logo />
      <div className="text text-remindersent">Reminder sent!</div>
      <div className="success-hand">
        <Hand />
      </div>
      <TimeoutNotification />
    </div>
  );
};

SuccessPage.propTypes = {
  history: PropTypes.object.isRequired
};

export default SuccessPage;
