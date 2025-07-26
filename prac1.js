// const fetch = require('node-fetch');
// const fs = require('fs');

//******Callbacks*********/

// One setTimeout
// setTimeout(() => {
//     console.log("Waited 1 second");
// }, 1000);

// // Nested setTimeouts
// setTimeout(() => {
//     console.log('3');
//     setTimeout(() => {
//         console.log('2');
//         setTimeout(() => {
//             console.log('1');
//         }) // defaults to 0 milliseconds meaning no delay
//     }, 1000);
// }, 1000);

// Button event handler in browser JavaScript
// const btn;
// btn.addEventListener('click', () => {

// })

// error first callback
// fs.readFile('./test.txt', { encoding: 'utf-8' }, (err, data) => {
//     if(err) {
//         console.error('ERROR');
//         console.error(err);
//     } else {
//         console.error('GOT DATA');
//         console.log(data);
//     }
// })



//******Promises*********/
// Create a promise
// const mypromise = new Promise((resolve, reject) => {
//     const rand = Math.floor(Math.random() * 2);
//     if(rand === 0) {
//         resolve();
//     } else {
//         reject();
//     }
// })
// mypromise
//     .then(() => console.log("Success"))
//     .catch(() => console.log("Something went wrong"));

// // fs readFile with promises
// fs.promises.readFile('./test.txt', { encoding: 'utf-8' })
//     .then((data) => console.log(data))
//     .catch((err) => console.log(err));async function getFetch(callback) {

// Async/Await
// const getFetched = async () => {
// try {
//     const response = await fetch('https://api.open-meteo.com/v1/forecast?latitude=0.34&longitude=32.57&current_weather=true')
//     const data = await response.json()
//     console.log(data);

// }catch(error){
//     console.log(error)
// }
// }
// getFetched()

// Promises
// const promise = new Promise((resolve, reject) => {
//     const xhr = new XMLHttpRequest();
//     const url = xhr.open('GET', 'https://api.open-meteo.com/v1/forecast?latitude=0.34&longitude=32.57&current_weather=true')
//     const request = url.send()
//     const response = request.setRequestHeader('json', 'application/json')

//     if (response) {
//         resolve()
//     } else {
//         reject()
//     }
// })

// promise
//     .then(() => console.log(response))
//     .catch((error) => console.error(error));

// // XMLHttpRequest
// const xhr = new XMLHttpRequest();
// xhr.open('GET', 'https://api.open-meteo.com/v1/forecast?latitude=0.34&longitude=32.57&current_weather=true')
// xhr.onload = function () {
//     if(xhr.status === 200) {
//         console.log(JSON.parse(xhr.responseText))
//     } else {
//         console.error('Error', xhr.status)
//     }
// }
// xhr.send()

// practice
// console.log(1);

// setTimeout(() => console.log(2));

// Promise.resolve().then(() => console.log(3));

// // The Promise schedules the setTimeout as a microtask but then after the console.log(4) is scheduled as a macrotask because of the setTimeout
// Promise.resolve().then(() => setTimeout(() => console.log(4)));

// Promise.resolve().then(() => console.log(5));

// setTimeout(() => console.log(6));

// console.log(7);

// Output
// 1, 7, 3, 5, 2, 6, 4
// async function fetchUrls () {
//     let urls = ['https://catfact.ninja/fact', 'https://dog.ceo/api/breeds/image/random', 'https://api.chucknorris.io/jokes/random'];

//     let controller = new AbortController();

//     let fetchJobs = urls.map(url => fetch(url, {
//         signal: controller.signal
//     }));
//     setTimeout(() => controller.abort(), 1000);

//     let results = await Promise.all(fetchJobs);
// }
// fetchUrls();

const nums = [1, 2, 3, 4, 5];

// const promise = (num) => {
//     return new Promise((resolve) => {
//         setTimeout(() => {
//             resolve(num);
//         }, 200);
//     });
// };
// nums.forEach((num) => {
//     promise(num).then((result) => {
//         console.log(result);
//     });
// });

// Consuming the promise using forEach loop
function display(numbers) {
    numbers.forEach(async (num) => {
      const dis = await promise(num);
      console.log(dis);
    });
  }
  
display(nums);