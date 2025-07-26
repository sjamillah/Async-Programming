// function outer() {
//     let count = 0;
//     return function inner() {
//       count++;
//       console.log(count);
//     };
//   }
//   const counter = outer();
//   counter();
//   counter();

// function createClickTracker() {
//     let clicks = 0;
//     return function () {
//       clicks++;
//       console.log(`Clicked ${clicks} times`);
//     };
//   }
  
//   for (var i = 0; i < 3; i++) {
//     setTimeout(() => console.log(i), 1000);
//   }

  for (let i = 0; i < 3; i++) {
    setTimeout(() => console.log(i), 1000);
  }
  