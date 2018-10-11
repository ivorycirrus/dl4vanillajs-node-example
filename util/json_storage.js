/* JSON file read/write module */
const fs = require("fs");
const jsonFile = {
	"read" : function(file){
		if(fs.existsSync(file)){
			let data = fs.readFileSync(file);
			return JSON.parse(data);
		}
	}
	,"write" : function(file, data){
		return fs.writeFileSync(file, JSON.stringify(data));
	}
};

module.exports = jsonFile;