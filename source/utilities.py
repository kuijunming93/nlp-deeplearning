import re

def cleanText(input):
    replacement = {
        r"i'm": "i am",
        r"he's": "he is",
        r"she's": "she is",
        r"that's": "that is",
        r"what's": "what is",
        r"where's": "where is",
        r"\'ll'": "will",
        r"\'ve'": "have",
        r"\'re'": "are",
        r"\'d'": "would",
        r"won't'": "will not",
        r"can't'": "cannot",
        r"[-()\"#/@;:{}+=~|.?,]": ""
    }
    output = input.lower()
    for k, v in replacement.items():
        output = re.sub(k, v, output)
    return output;

def consolidateWordCount(inputList):
    output = {}
    for element in inputList:
        for sentences in element:
            for word in sentences.split():
                if word not in output:
                    output[word] = 1
                else:
                    output[word] += 1
    return output

def convertVocabToInt(vocab, threshold):
    output = {}
    count = 0
    for word, count in vocab.items():
        if count > threshold:
            output[word] = count
            count += 1

    tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
    for token in tokens:
        output[token] = len(output) + 1
    return output

def convertSentencesToInt(cleaned_input, vocab):
    output = []
    for sentence in cleaned_input:
        ints = []
        for word in sentence.split():
            if word in vocab:
                ints.append(vocab[word])
            else:
                ints.append(vocab["<OUT>"])
        output.append(ints)
    return output


