// Write a JavaScript function that uses a chain of .then() calls to perform a series of asynchronous tasks.

// function promiseChain() {
//     return fetch('https://api.chucknorris.io/jokes/random')
//             .then((response) => response.json())
//             .then((data) => data.value)
// }
// promiseChain()
//     .then((joke) => console.log(joke))
//     .catch((err) => console.log(err))


// function promiseChain(url) {
//     return new Promise((resolve, reject) => {
//         const reader = new FileReader()
//         reader.onload = () => resolve("Image loaded")
//         reader.onerror = () => reject("Image failed to load")
//         reader.readAsText(file)
//     })
// }
// promiseChain('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.istockphoto.com%2Fphotos%2Fcreative-arts&psig=AOvVaw2-WIY0ULB-nF6eN9UqgVAU&ust=1749334987645000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCIjvtuTq3Y0DFQAAAAAdAAAAABAE')
//     .then((data) => console.log(data))
//     .catch((err) => console.log(err))

function promiseChain(task, delay) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(`${task} completed`)
    }, delay)
})
}
promiseChain('Task 1', 1000)
    .then((result1) => {
        console.log(result1)
        return promiseChain('Task 2', 2000)
    })
    .then((result2) => {
        console.log(result2)
        return promiseChain('Task 3', 3000)
    })
    .then((result3) => {
        console.log(result3)
        console.log("All tasks completed")
    })