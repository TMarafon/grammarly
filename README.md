---
title: Grammarly
emoji: 🐨
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.1.1
app_file: app.py
pinned: false
license: mit
---

# Grammarly Project

## Overview

This project is an advanced English grammar correction and practice tool built using OpenAI's language models. It is designed to help users enhance their English skills by providing text correction, detailed explanations, and interactive grammar practice.

## Features

- **Text Correction**: Input text for review and correction. The system returns the corrected text, an overall score, and a detailed explanation of the corrections.
- **Contextual Analysis**: Add context to your text for more accurate and tailored corrections.
- **Sentence-by-Sentence Analysis**: Breakdown of each sentence in the input text, including reviews, comments, and individual scores.
- **Practice Sessions**: Interactive sessions for practicing grammar by correcting sentences with similar issues.
- **Audio Output**: Convert corrected text to speech, aiding in pronunciation and listening skills.

## Getting Started

1. **Install Dependencies**: Ensure Python is installed and install required packages like `gradio`, `pandas`, `openai`, etc.
2. **API Key Configuration**: Set up your OpenAI API key in the application.
3. **Launch Application**: Run the Python script to start the Gradio interface.
4. **Use the Tool**: Input your text and context for analysis and correction.

## Usage

- **Text Input**: Enter the text you want corrected in the 'Input Text' tab.
- **Context Input**: Provide the context for your text in the 'Context' tab.
- **Review and Feedback**: Post submission, the system provides the corrected version, explanation, score, and a detailed breakdown per sentence.
- **Interactive Practice**: Select a sentence and start a practice session to correct a similar sentence with grammar issues.

## Additional Features

- **Text to Voice**: Hear how the corrected text sounds by converting it to speech.
- **API Key Update**: Update the OpenAI API key directly through the interface.

## License

This project is released under the [MIT License](LICENSE.md).

