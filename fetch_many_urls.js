async function fetchUrls() {
    let urls = ['https://catfact.ninja/fact', 'https://dog.ceo/api/breeds/image/random', 'https://api.chucknorris.io/jokes/random'];

    let controller = new AbortController();
    let signal = controller.signal;

    try {
        setTimeout(() => controller.abort(), 3560);
        let fetchdata = urls.map(url => 
            fetch(url, { signal })
                .then(response => response.json())   // fetches each url from the list and the response is got in json format     
        );
        let results = await Promise.all(fetchdata);
        console.log(results);
    }
    catch(err) {
        if(err.name == 'AbortError') {
            console.log('Aborted');
        } else {
            throw err;
        }
    }
}
fetchUrls();