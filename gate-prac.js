// Write a function called fetchWithCallback that accepts a callback function as an argument. 
// The function should return a Promise that attempts to simulate fetching data 
// (you can use setTimeout to mimic a delay). 
// If the operation is successful, resolve the Promise and call the callback with the data. 
// If it fails, reject the Promise and call the callback with an error."

async function fetchWithCallback(callback, url) {
    try {
        let data = await fetch(url)
        let response = await data.json()
        callback(null, response)
    } catch(err) {
        callback(err, null)
    }
}

function callback(err, data) {
    if(err) {
        console.log("Failed to fetch", err.message)
    }
    else {
        console.log("Fetched data successfully", data)
    }
}

fetchWithCallback(callback, 'https://api.chucknorris.io/jokes/random')

// function fetchWithCallback(callback) {
//     return new Promise((resolve, reject) => {
//         setTimeout(() => {
//             const isSuccess = Math.random() < 0.7

//             if(isSuccess) {
//                 const data = { message: "Data fetched successfully"}
//                 callback(null, data)
//                 resolve(data)
//             } else {
//                 const error = new Error("Failed to fetch the data")
//                 callback(error, null)
//                 reject(error)
//             }
//         }, 1000)
//     })
// }

// function callback(err, data) {
//     if(err) {
//         console.log(err.message)
//     } else {
//         console.log(data)
//     }
// }

// fetchWithCallback(callback)
//     .then((data) => console.log(data))
//     .catch((err) => console.log(err.message))