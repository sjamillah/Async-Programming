function getCallback(callback) {
    setTimeout(callback, 2000)
}

function callback() {
    console.log("Hello World")
}

getCallback(callback)