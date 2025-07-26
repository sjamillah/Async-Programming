// Write a JavaScript program that takes an array of Promises and logs both resolved and rejected results using Promise.allSettled().

const promise1 = new Promise((resolve) => {
    setTimeout(() => resolve('meet us'), 1000)
})

const promise2 = new Promise((resolve, reject) => {
    setTimeout(() => reject('failed'), 1000)
})

const promise3 = new Promise((resolve) => {
    setTimeout(() => resolve('new me'), 500)
})

Promise.allSettled([promise1, promise2, promise3])
    .then((data) => console.log(data))
    .catch((err) => err.message)