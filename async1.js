let polling = true;
let timerId = null;

// Simulated fetch from an API
async function fetchStockPrices() {
  try {
    console.log("Fetching stock prices...");
    // Replace with real fetch: await fetch("https://api.example.com/stocks");
    await new Promise((resolve) => setTimeout(resolve, 1000)); // simulate network delay
    console.log("Prices updated at", new Date().toLocaleTimeString());
  } catch (error) {
    console.error("Error fetching stock prices:", error);
  }
}

// Polling function
async function startPolling() {
  while (polling) {
    await fetchStockPrices();
    if (polling) {
      await new Promise((resolve) => setTimeout(resolve, 2000)); // wait before next fetch
    }
  }
}

// Start polling when page is visible
function handleVisibilityChange() {
  if (document.hidden) {
    polling = false;
    console.log("Paused polling (tab hidden)");
  } else {
    if (!polling) {
      polling = true;
      console.log("Resumed polling (tab visible)");
      startPolling(); // restart
    }
  }
}

document.addEventListener("visibilitychange", handleVisibilityChange);

// Initial start
startPolling();