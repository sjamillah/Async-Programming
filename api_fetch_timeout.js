async function fetchApi(url, timeout) {
        // let data = await fetch(url)
        // let response = await data.json()
        // setTimeout(() => {
        //     console.log(response)
        // }, 1000)
        let controller = new AbortController()
        let signal = controller.signal

        let timerId = setTimeout(() => {
                controller.abort()
            }, timeout)

        try {
            let data = await fetch(url, { signal })
            clearTimeout(timerId)
            let response = await data.json()
            console.log("Fetched successfully", response)
        } catch(err) {
            if(err == 'AbortError'){
                console.log("Aborted")
            } else {
                console.log("Failed to fetch the data", err.message)
            }
        }
}
fetchApi('https://api.chucknorris.io/jokes/random', 1000)