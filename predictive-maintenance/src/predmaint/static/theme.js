// Apply a light/dark choice before first paint (kept out of the page so the CSP can stay strict).
// Priority: ?theme=light|dark in the URL, then the saved choice, then the OS setting.
(function () {
  var choice = new URLSearchParams(window.location.search).get("theme");
  if (choice !== "light" && choice !== "dark") {
    try {
      choice = localStorage.getItem("pm-theme");
    } catch (e) {
      choice = null; /* storage disabled: follow the OS setting */
    }
  }
  if (choice === "light" || choice === "dark") document.documentElement.dataset.theme = choice;
})();
