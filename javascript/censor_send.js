// Auto-Censor: activate the "🔞 Censor" tab.
// Called by Forge's copypaste infra via `paste_button.click(fn=None, _js="switch_to_auto_censor")`
// (the JS name is derived from the ParamBinding tabname "auto_censor").
// Robust against tab reordering: matches the nav button by its label text instead of a fixed index.
function switch_to_auto_censor() {
    const root = gradioApp();
    let navButtons = root.querySelectorAll("#tabs > div.tab-nav > button");
    if (!navButtons || navButtons.length === 0) {
        navButtons = root.querySelectorAll("#tabs button");
    }
    for (const btn of navButtons) {
        if (btn.textContent && btn.textContent.indexOf("Censor") !== -1) {
            btn.click();
            break;
        }
    }
    return Array.from(arguments);
}
