// Write a JavaScript function that ensures only a specified number of asynchronous requests are made simultaneously.

async function throt(urls, reqlimittime) {
    const results = [];

    for(let index = 0; index < urls.length; index += reqlimittime) {
        const batch = urls.slice(index, index + reqlimittime);
        const batchRes = await Promise.all(
            batch.map(async url => {
                try {
                    const res = await fetch(url)
                    return await res.json()
                } catch (err) {
                    return { error: err.message }
                }
            })
        );
        results.push(...batchRes);
    }
    return results;
}

const urls = ['https://www.boredapi.com/api/activity', 'https://api.chucknorris.io/jokes/random', 'https://catfact.ninja/fact']
throt(urls, 2).then(data => console.log(data))