// Write a JavaScript function that fetches data from an API and retries the request a specified number of times if it fails. You should log to the console “Retrying….” when retrying the request.
async function retrial(url, maxRetries) {
    for(let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            let response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP Error with status: ${response.status}`)
            
            let data = await response.json();
            console.log(`Data fetched successfully ${JSON.stringify(data, null, 2)}`);
            break; // stops retrying once it is successful
        } catch(err) {
            if(attempt < maxRetries) {
                console.log(`Retrying... (${attempt})`);
            } else {
                console.log('All retries failed.');
            }
        }
    }
}
retrial('https://jsonplaceholder.typicode.com/posts', 3);