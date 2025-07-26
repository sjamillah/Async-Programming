// async function* myAsyncGenerator() {
//   yield await Promise.resolve(1);
//   yield await Promise.resolve(2);
//   yield await Promise.resolve(3);
// }

// // (async () => {
// //   for await (const num of myAsyncGenerator()) {
// //     console.log(num);
// //   }
// // })();
// let gen = myAsyncGenerator()
// console.log(gen)
async function fetchData() {
  try {
    const result = await someAsyncOperation();
  } catch (err) {
    console.error(err.stack);
  }
}
