// Write a javascript function that displays a number every two seconds and stops displaying after 5 seconds
// input: number
// output: same number every 2 seconds displayed and stops displaying after 5 seconds

// function displayNum(num) {
//     setTimeout(() => {
//         console.log(num)
//         setTimeout(() => {
//             abort()
//         }, 5000)
//     }, 2000)   
// }

// displayNum(3)


function displayNum() {
    let rand = Math.floor(Math.random() % 2);
    let timerId = setInterval(() => {
        let newNum = rand++;
        console.log(newNum);
    }, 2000);
    setTimeout(() => clearInterval(timerId), 5000);
}
displayNum();

// // Aborting
// setTimeout(() => {
//     console.log('2')
// })

