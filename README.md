# SMS & Phishing Spam Detector (NLP Deep Learning)

A Deep Learning natural language processing (NLP) application classifying incoming text messages and emails into Legitimate (Ham) or Spam/Phishing threats.

## Overview
* **Domain:** Natural Language Processing (NLP) / Cybersecurity
* **Architecture:** Multi-Layer Perceptron (MLP) with Keras `TextVectorization` and Dropout regularization
* **Framework:** Streamlit, TensorFlow (Keras), NumPy
* **Objective:** Binary classification of text threats (`0: Ham`, `1: Spam`)

## Features
* Live interactive text area to paste raw SMS or email content.
* Pre-built buttons to quickly test both legitimate and phishing templates.
* Visual risk-meter with percentage probability.
