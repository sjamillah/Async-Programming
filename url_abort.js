// Write a function that takes a URL and fetches data from that URL but aborts when the request takes more than 10ms
// input: url

// function fetchData(url) {
//     let timer = 10;
//     setInterval(async () => {
//         const response = await fetch(url)
//         const data = await response.json()
//         console.log(data)
//         setTimeout(() => {
//             console.log('Timeout')
//         }, timer++)
//     }, timer)
// }

// fetchData('https://api.open-meteo.com/v1/forecast?latitude=0.34&longitude=32.57&current_weather=true')

async function fetchData(url) {
    let controller = new AbortController();
    let signal = controller.signal;

    try {
        setTimeout(() => controller.abort(), 10)
        let response = await fetch(url, { signal });
        let data = await response.json();
        console.log(data);
    } catch(err) {
        if(err.name == 'AbortError') console.log('The content couldn\'t be fetched');
        else throw err;
    }
}
fetchData('https://catfact.ninja/fact');
