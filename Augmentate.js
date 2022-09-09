function augmentFunc(sample) {
    const img = sample.image;
    const augmentedImg = randomRotate(
        randomSkew(randomMirror(img)));

    return {image: augmentedImg, label: sample.label};
}

const {trainingDataSet, validationDataset} = getDatsetsFromSource();
augmentedDataset = trainingDataSet.repeat().map(augmentFn).batch(BATCH_SIZE);

