// Write a JavaScript function that returns a Promise that resolves with a "Hello, World!" message after 1 second.

function promiseHolder() {
    return new Promise((resolve) => {
        setTimeout(() => 
        resolve("Hello World!"), 1000)
    })
}
// We use .then() to access and display the result of the promise.
// Without .then(), the resolved value would not be shown or used.
promiseHolder()
    .then((message) => console.log(message))