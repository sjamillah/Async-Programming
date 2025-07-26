// Write a JavaScript function that takes multiple Promises and resolves with the first successful result using Promise.any().

const promise1 = new Promise((resolve) => {
    setTimeout(() => resolve('meet us'), 1000)
})

const promise2 = new Promise((resolve, reject) => {
    setTimeout(() => reject('failed'), 1000)
})

const promise3 = new Promise((resolve) => {
    setTimeout(() => resolve('new me'), 500)
})

Promise.any([promise1, promise2, promise3])
    .then((data) => console.log(data))
    .catch((err) => err.message)