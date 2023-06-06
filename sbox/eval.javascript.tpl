const fs = require('fs');
let script = fs.readFileSync('/answer');
try {
  script += '\\nvar _rv = {{call}};'
  eval(script);
} catch(error) {
  _rv = { error: error.message, exc_type: error.name }
}
console.log('###' + JSON.stringify(_rv));