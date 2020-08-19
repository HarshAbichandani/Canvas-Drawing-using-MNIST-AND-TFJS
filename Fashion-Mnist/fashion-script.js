import {FMnistData} from './fashion-data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;

function getModel() {
    
    
    
    // In the space below create a convolutional neural network that can classify the 
    // images of articles of clothing in the Fashion MNIST dataset. Your convolutional
    // neural network should only use the following layers: conv2d, maxPooling2d,
    // flatten, and dense. Since the Fashion MNIST has 10 classes, your output layer
    // should have 10 units and a softmax activation function. You are free to use as
    // many layers, filters, and neurons as you like.  
    // HINT: Take a look at the MNIST example.
    model = tf.sequential();
    
    // YOUR CODE HERE
// ***  tf.layers.conv2d creates a cross related kernel with layer i/p and a tensor of o/p . filters is output dimensionality. kernels are layers
    model.add(tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.flatten());
	
    //Hidden layer of 128
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	// o/p of 10 units
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
    
    // Compile the model using the categoricalCrossentropy loss,
    // the tf.train.adam() optimizer, and accuracy for your metrics.
    model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']});
        
        // YOUR CODE HERE
    
    return model;
}

async function train(model, data) {
        
    // Set the following metrics for the callback: 'loss', 'val_loss', 'acc', 'val_acc'.
    const metrics = ['loss','val_loss','acc','val_acc'];// YOUR CODE HERE    

        
    // Create the container for the callback. Set the name to 'Model Training' and 
    // use a height of 1000px for the styles. 
    const container = {name: 'Model Training', styles: {height: '1000px'}}; // YOUR CODE HERE   
    
    
    // Use tfvis.show.fitCallbacks() to setup the callbacks. 
    // Use the container and metrics defined above as the parameters.
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics); // YOUR CODE HERE
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 6000;
    const TEST_DATA_SIZE = 1000;
    
    // Get the training batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28,28,1]),
            d.labels
        ];
    }); // YOUR CODE HERE

    
    // Get the testing batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE,28,28,1]),
            d.labels
        ];
    }); // YOUR CODE HERE

    
    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        //fitcallbacks declared up will show the callbacks with metrics declared
        callbacks: fitCallbacks
    });
}

function setPosition(e){
    pos.x = e.clientX-100;
    pos.y = e.clientY-100;
}
//    draw func manages the drawing , oving etc of mouse cursor on canvas
function draw(e) {
    if(e.buttons!=1) return;
    ctx.beginPath();
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    rawImage.src = canvas.toDataURL('image/png');
}
    
function erase() {
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,280,280);
}
    
function save() {
    // rawImage is passed to html and 1 here defines monochrome color i.e. black and white
    var raw = tf.browser.fromPixels(rawImage,1);
    var resized = tf.image.resizeBilinear(raw, [28,28]);
    var tensor = resized.expandDims(0);
    
    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync();
    
    var classNames = ["T-shirt/top", "Trouser", "Pullover", 
                      "Dress", "Coat", "Sandal", "Shirt",
                      "Sneaker",  "Bag", "Ankle boot"];
            
            
    alert(classNames[pIndex]);
}
  
//init sets up the UI i.e. the canvas to draw, etc
function init() {
    canvas = document.getElementById('canvas');
    rawImage = document.getElementById('canvasimg');
    ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,280,280);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mousedown", setPosition);
    canvas.addEventListener("mouseenter", setPosition);
    saveButton = document.getElementById('sb');
    // save declared up
    saveButton.addEventListener("click", save);
    clearButton = document.getElementById('cb');
    // erase declared above
    clearButton.addEventListener("click", erase);
}


async function run() {
    
    const data = new FMnistData();
    
    //this will download the sprite img and get it in the memory
    await data.load();
    
    //defined above
    const model = getModel();
    
    // layers look like described
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);

    //train func defined above
    await train(model, data);
    await model.save('downloads://my_model');
    init();
    alert("Training is done, try classifying your drawings!");
}
// this means as soon as document is loaded , it will call the run function
document.addEventListener('DOMContentLoaded', run);