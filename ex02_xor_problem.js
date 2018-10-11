const dl = require("dl4vanillajs");
const storage = require("./util/json_storage");
const FILE_PRE_TRAINED = "./pre_trained/ex02_pretrained_weights.json";
const FILE_OUTPUT_TRAINED = null;

// input data
x_data = [[0,0],[0,1],[1,0],[1,1]];
y_data = [[0],[1],[1],[0]];

// Params
const NUM_OF_KINDS = 2;
const NUM_OF_HIDDEN_LAYERS = 4;
const NUM_OF_OUTPUTS = 1;

const BATCH_SIZE = 4;
const LEARNING_RATE = 0.01;
const TRAIN_LIMIT = 2001;
const LOG_INTERVAL = 200;

// artificial neural nets with 3 layers
let MultiLayerNet = function(input_size, hidden_size, output_size){
	let thiz = this;

	if(!(thiz.params = storage.read(FILE_PRE_TRAINED))) {
		thiz.params = {
			'W1' : dl.mat.matrix([input_size, hidden_size], x=>(Math.random()*10.0-5.0)),
			'b1' : dl.mat.matrix([1,hidden_size], 0),
			'W2' : dl.mat.matrix([hidden_size, hidden_size], x=>(Math.random()*10.0-5.0)),
			'b2' : dl.mat.matrix([1,hidden_size], 0),
			'Wout' : dl.mat.matrix([hidden_size, output_size], x=>(Math.random()*10.0-5.0)),
			'bout' : dl.mat.matrix([1,output_size], 0)
		};
	}

	// forward process
	thiz.predict = function(x){
		// layer 1
		let L1 = dl.mat.mul(x, thiz.params['W1']);
		L1 = dl.mat.add(L1, thiz.params['b1']);
		L1 = dl.actv.sigmoid(L1);
		// layer 2
		let L2 = dl.mat.mul(L1, thiz.params['W2']);
		L2 = dl.mat.add(L2, thiz.params['b2']);
		L2 = dl.actv.sigmoid(L2);
		// output layer
		let Lout = dl.mat.mul(L2, thiz.params['Wout']);
		Lout = dl.mat.add(Lout, thiz.params['bout']);
		Lout = dl.actv.sigmoid(Lout);
		// output
		return Lout;
	};

	// Loss function
	thiz.loss = function(x, t){		
		let y = thiz.predict(x);
		return dl.loss.cross_entropy_with_logits(y, t);
	};

	// Train weights and biases
	thiz.train = function(x, t, batch_size){
		for(let b = 0 ; b < batch_size ; b++){
			let _x = x.slice(b,b+1);
			let _t = t.slice(b,b+1);
			for(i in thiz.params) {
				thiz.params[i] = dl.opt.gradient_decent_optimizer(()=>thiz.loss(_x,_t), thiz.params[i], LEARNING_RATE);
			}
		}
	};
};

// Start Training
let nn = new MultiLayerNet(NUM_OF_KINDS, NUM_OF_HIDDEN_LAYERS, NUM_OF_OUTPUTS);

console.log("==[TRAIN]==")
for(let step = 0 ; step < TRAIN_LIMIT ; step++) {
	// train
	for(var k = 0 ; k < x_data.length/BATCH_SIZE ; k += BATCH_SIZE){
		let x = x_data.slice(k,k+BATCH_SIZE);
		let t = y_data.slice(k,k+BATCH_SIZE);
		
		nn.train(x, t, BATCH_SIZE);		
	}

	// loss
	if(step%LOG_INTERVAL == 0) {
		let arrLoss = [];
		for(let i = 0 ; i < x_data.length ; i++) arrLoss.push(nn.loss(x_data.slice(i,i+1), y_data.slice(i,i+1)));
		console.log("step : "+step+" loss : "+dl.mat.reduce_mean(arrLoss));	
	}
}

// Test
console.log("==[TEST]==")
for(let i = 0 ; i < x_data.length ; i++) {
	let pred = dl.mat.flat(nn.predict(x_data.slice(i,i+1)))[0];
	let correct = y_data[i][0];
	console.log(`Prediction : ${pred.toFixed(2)} \tCorrect : ${correct.toFixed(2)}`);
}

// Save weights
if(FILE_OUTPUT_TRAINED) storage.write(FILE_PRE_TRAINED, nn.params);