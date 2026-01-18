from colorama import Back, Fore, Style, init

# Initialize colorama
init(autoreset=True)


def pretty_print_mapping(mapping_dict, indent=0):
    """
    Pretty print the feature mapping dictionary with colors.

    Parameters:
    -----------
    mapping_dict : dict
        Dictionary mapping new features to lists of old features
    indent : int
        Current indentation level (used for recursion)
    """

    indent_str = "  " * indent

    for key, value in mapping_dict.items():
        if key == "__unmapped__":
            # Special handling for unmapped
            if isinstance(value, list) and value:
                print(f"{indent_str}{Fore.YELLOW}⚠ {key}:{Style.RESET_ALL}")
                for item in value:
                    print(f"{indent_str}  {Fore.YELLOW}• {item}{Style.RESET_ALL}")
            elif isinstance(value, list):
                print(f"{indent_str}{Fore.YELLOW}⚠ {key}: []{Style.RESET_ALL}")
            continue

        if isinstance(value, dict):
            # Section header (like "Assets", "Current", etc.)
            if indent == 0:
                print(f"\n{Fore.CYAN}{Back.BLACK}{'=' * 80}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{Back.BLACK} {key.upper()} {Style.RESET_ALL}")
                print(f"{Fore.CYAN}{Back.BLACK}{'=' * 80}{Style.RESET_ALL}\n")
            else:
                print(f"{indent_str}{Fore.MAGENTA}▼ {key}{Style.RESET_ALL}")

            # Recurse into nested dictionary
            pretty_print_mapping(value, indent + 1)

        elif isinstance(value, list):
            # Leaf node - show mapping
            if value:
                # Has mappings
                print(f"{indent_str}{Fore.GREEN}✓ {key}:{Style.RESET_ALL}")
                for old_feature in value:
                    print(
                        f"{indent_str}  {Fore.WHITE}├─ {old_feature}{Style.RESET_ALL}"
                    )
            else:
                # Empty list - no mappings
                print(
                    f"{indent_str}{Fore.RED}✗ {key}: {Fore.LIGHTBLACK_EX}[no mapping]{Style.RESET_ALL}"
                )


def print_mapping_summary(mapping_dict):
    """
    Print a summary of the mapping statistics.

    Parameters:
    -----------
    mapping_dict : dict
        Dictionary mapping new features to lists of old features
    """

    def count_mappings(d):
        """Recursively count new feature categories and old features."""
        mapped_new_features = 0  # New categories with mappings
        unmapped_new_features = 0  # New categories without mappings
        mapped_old_features = set()  # Old features that were mapped
        unmapped_old_features = set()  # Old features in __unmapped__

        for key, value in d.items():
            if key == "__unmapped__":
                if isinstance(value, list):
                    unmapped_old_features.update(value)
                continue

            if isinstance(value, dict):
                # Recurse into nested structure
                m_new, u_new, m_old, u_old = count_mappings(value)
                mapped_new_features += m_new
                unmapped_new_features += u_new
                mapped_old_features.update(m_old)
                unmapped_old_features.update(u_old)
            elif isinstance(value, list):
                # Leaf node: new feature category
                if value:
                    mapped_new_features += 1
                    mapped_old_features.update(value)
                else:
                    unmapped_new_features += 1

        return (
            mapped_new_features,
            unmapped_new_features,
            mapped_old_features,
            unmapped_old_features,
        )

    mapped_new, unmapped_new, mapped_old, unmapped_old = count_mappings(mapping_dict)

    print(f"\n{Fore.CYAN}{Back.BLACK}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Back.BLACK} MAPPING SUMMARY {Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Back.BLACK}{'=' * 80}{Style.RESET_ALL}\n")

    # New feature categories summary
    total_new = mapped_new + unmapped_new
    print(
        f"{Fore.WHITE}New Feature Categories: {Fore.CYAN}{total_new}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.GREEN}  ✓ Mapped: {mapped_new} "
        f"({100 * mapped_new / total_new:.1f}%){Style.RESET_ALL}"
    )
    print(
        f"{Fore.RED}  ✗ Unmapped: {unmapped_new} "
        f"({100 * unmapped_new / total_new:.1f}%){Style.RESET_ALL}"
    )

    # Old feature fields summary
    total_old = len(mapped_old) + len(unmapped_old)
    print(f"\n{Fore.WHITE}Old Feature Fields: {Fore.CYAN}{total_old}{Style.RESET_ALL}")
    print(
        f"{Fore.GREEN}  ✓ Mapped: {len(mapped_old)} "
        f"({100 * len(mapped_old) / total_old:.1f}%){Style.RESET_ALL}"
    )
    print(
        f"{Fore.RED}  ✗ Unmapped: {len(unmapped_old)} "
        f"({100 * len(unmapped_old) / total_old:.1f}%){Style.RESET_ALL}"
    )

    # Show unmapped fields if there are any
    if unmapped_old:
        unmapped_list = sorted(list(unmapped_old))
        print(f"{Fore.YELLOW}    Fields: {unmapped_list}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")


def pretty_print_full_mapping(mapping_dict, show_summary=True, statement_type=""):
    """
    Complete pretty print with optional summary.

    Parameters:
    -----------
    mapping_dict : dict
        Dictionary mapping new features to lists of old features
    show_summary : bool
        Whether to show summary statistics at the end
    """

    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'#' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}#{' ' * 78}{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}{Style.BRIGHT}#  {statement_type.upper()}: OLD → NEW{' ' * 48}{Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}{Style.BRIGHT}#{' ' * 78}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'#' * 80}{Style.RESET_ALL}")

    pretty_print_mapping(mapping_dict)

    if show_summary:
        print_mapping_summary(mapping_dict)
